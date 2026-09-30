"""双塔召回模型：用户塔 + 物品塔 + in-batch sampled softmax（流行度 logQ 修正）。

工业界双塔召回的标准配方（对照 docs/系统设计.md 第 4 节）：
1. 两侧各自独立编码，只在最后做内积 —— 训练形态必须与线上检索形态一致，
   否则「训练用交叉特征、线上只有向量内积」会造成 train/serve 偏差；
2. batch 内其他用户的正样本当作 in-batch 负样本：batch=1024 时一次前向
   同时完成 1024 个正例和 1024×1023 个负例的更新，性价比极高；
3. logQ 修正：热门物品天然更常出现在 batch 里被当负例，要按其出现概率
   把 logits 减去 log(q) 补偿，否则长尾物品的向量被系统性推远；
4. 物品塔可选接入内容语义向量（bge-small 输出），让零行为新物品也有可用向量，
   这是冷启动内容通道成立的前提。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TwoTowerConfig:
    num_users: int
    num_items: int
    num_genres: int
    num_gender: int
    num_age: int
    num_occ: int
    id_dim: int = 64
    hidden: Sequence[int] = (128, 64)
    out_dim: int = 64
    dropout: float = 0.1
    use_content: bool = False
    content_dim: int = 0
    content_proj_dim: int = 32


def _masked_mean(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按 mask 加权均值；mask 全 0 时返回全 0 向量（区别于把垃圾 embedding 平均进来）。"""
    mask = mask.to(emb.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (emb * mask).sum(dim=1) / denom


class _Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int, dropout: float):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TwoTower(nn.Module):
    def __init__(self, cfg: TwoTowerConfig, content: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.id_dim
        # ---------------- 用户塔 ----------------
        self.user_emb = nn.Embedding(cfg.num_users, d)
        self.gender_emb = nn.Embedding(cfg.num_gender, d // 2)
        self.age_emb = nn.Embedding(cfg.num_age, d // 2)
        self.occ_emb = nn.Embedding(cfg.num_occ, d // 2)
        self.hist_emb = nn.Embedding(cfg.num_items, d, padding_idx=0)
        # user_dense 维度在构造时未知（LazyLinear 首次前向时物化）
        self.user_dense_proj = nn.LazyLinear(d)
        # 输入 = user_emb(d) + 历史均值(d) + 三个人口画像(d//2 × 3) + dense 投影(d)
        user_in = d * 3 + (d // 2) * 3
        self.user_tower = _Tower(user_in, list(cfg.hidden), cfg.out_dim, cfg.dropout)

        # ---------------- 物品塔 ----------------
        self.item_emb = nn.Embedding(cfg.num_items, d)
        self.genre_emb = nn.Embedding(cfg.num_genres + 1, d // 2, padding_idx=0)
        self.item_dense_proj = nn.LazyLinear(d)
        # 输入 = item_emb(d) + 类目均值(d//2) + dense 投影(d)
        item_in = d * 2 + d // 2
        if cfg.use_content and cfg.content_dim > 0:
            self.content_proj = nn.Linear(cfg.content_dim, cfg.content_proj_dim)
            self.register_buffer(
                "content", torch.as_tensor(content, dtype=torch.float32)
                if content is not None else torch.zeros(cfg.num_items, cfg.content_dim))
            item_in += cfg.content_proj_dim
        else:
            self.content = None
        self.item_tower = _Tower(item_in, list(cfg.hidden), cfg.out_dim, cfg.dropout)

        # 可学习温度：logits = (u·v) * logit_scale，训练中自适应召回难度
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.05)

    # ---------------------------------------------------------- 用户塔
    def user_vec(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        d = self.cfg.id_dim
        hist = torch.as_tensor(batch["hist_item_idx"], dtype=torch.long)
        hist_mask = (hist >= 0)
        hist_idx = hist.clamp(min=0)  # padding 位置读到 0 号向量，随后被 mask 掉
        hist_vec = _masked_mean(self.hist_emb(hist_idx), hist_mask)
        feats = torch.cat([
            self.user_emb(batch["user_idx"]),
            hist_vec,
            self.gender_emb(batch["gender_idx"]),
            self.age_emb(batch["age_idx"]),
            self.occ_emb(batch["occ_idx"]),
            self.user_dense_proj(batch["user_dense"].to(torch.float32)),
        ], dim=-1)
        return F.normalize(self.user_tower(feats), dim=-1)

    # ---------------------------------------------------------- 物品塔
    def item_vec(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        d = self.cfg.id_dim
        genre_ids = torch.as_tensor(batch["genre_ids"], dtype=torch.long)
        genre_mask = torch.as_tensor(batch["genre_mask"], dtype=torch.long)
        genre_vec = _masked_mean(self.genre_emb(genre_ids), genre_mask)
        parts = [
            self.item_emb(batch["item_idx"]),
            genre_vec,
            self.item_dense_proj(batch["item_dense"].to(torch.float32)),
        ]
        if self.content is not None:
            c = self.content[batch["item_idx"].to(self.content.device)]
            parts.append(self.content_proj(c.to(torch.float32)))
        feats = torch.cat(parts, dim=-1)
        return F.normalize(self.item_tower(feats), dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.user_vec(batch), self.item_vec(batch)

    def n_params(self) -> int:
        # Lazy 模块（LazyLinear 等）在首次 forward 前参数是
        # UninitializedParameter，numel() 会报错——统计时跳过即可
        from torch.nn.parameter import UninitializedParameter
        return sum(p.numel() for p in self.parameters()
                   if not isinstance(p, UninitializedParameter))


class SampledSoftmaxLoss(nn.Module):
    """in-batch sampled softmax + logQ 修正 + 假负例屏蔽。

    logits[i, j] = u_i · v_j * scale / tau - log(q_j)
    其中 q_j 是物品 j 的边际出现概率（流行度）。对角线为正样本。
    另外把「batch 里恰好与正样本同 id 的列」从负例里屏蔽掉（假负例），
    避免 1024 大小时小物品库上的重复惩罚。
    """

    def __init__(self, item_pop: torch.Tensor, tau: float = 0.05):
        super().__init__()
        self.register_buffer("log_pop", torch.log(item_pop.to(torch.float32).clamp(min=1e-8)))
        self.tau = float(tau)

    def forward(
        self,
        u_vec: torch.Tensor,
        i_vec: torch.Tensor,
        item_idx: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.clamp(logit_scale, 0.1, 50.0) / self.tau
        logits = u_vec @ i_vec.t() * scale
        logits = logits - self.log_pop[item_idx].unsqueeze(0)  # logQ 修正
        pos = torch.arange(len(item_idx), device=item_idx.device)
        # 假负例：与该行正样本同 id 的其他列
        dup = (item_idx.unsqueeze(0) == item_idx.unsqueeze(1))
        dup.fill_diagonal_(False)
        logits = logits.masked_fill(dup, float("-inf"))
        return F.cross_entropy(logits, pos)
