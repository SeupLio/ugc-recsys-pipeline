"""精排模型：DeepFM / DIN / ESMM，同输入对照实验。

三种模型吃完全相同的特征 batch（recsys.data.torch_data.build_batch 组装），
保证 AUC 差距可以干净归因到「模型结构」而不是特征差异：
- DeepFM：FM 显式二阶交叉 + DNN 隐式高阶交叉，共享同一份 embedding；
- DIN：在 DeepFM 基础上把用户历史行为用 target attention 池化，
  「与候选物品相关的历史」获得更大权重（这是它在推荐场景赢过 DeepFM 的原因）；
- ESMM：多目标（CTR + CVR）全空间建模，解决 CVR 的样本选择偏差（SSB）
  与数据稀疏（DS）：pCTCVR = pCTR × pCVR，两个塔共享底层 embedding。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RankConfig:
    num_users: int
    num_items: int
    num_genres: int
    num_gender: int
    num_age: int
    num_occ: int
    id_dim: int = 16
    hidden: Sequence[int] = (64, 32)
    dropout: float = 0.1
    max_seq_len: int = 50      # 仅 DIN 使用
    # 注意：FM 的二阶交叉要求所有字段 embedding 同维（公式里的 v_f 必须可两两内积），
    # 因此人口画像 / 类目字段也统一用 id_dim；profile_dim / genre_dim 保留为兼容字段。


class _SharedEmbedding(nn.Module):
    """三个精排模型共享的字段 embedding 层（DeepFM 的「共享 embedding」设计）。"""

    def __init__(self, cfg: RankConfig):
        super().__init__()
        d = cfg.id_dim
        self.user_emb = nn.Embedding(cfg.num_users, d)
        self.item_emb = nn.Embedding(cfg.num_items, d)
        self.gender_emb = nn.Embedding(cfg.num_gender, d)
        self.age_emb = nn.Embedding(cfg.num_age, d)
        self.occ_emb = nn.Embedding(cfg.num_occ, d)
        self.genre_emb = nn.Embedding(cfg.num_genres + 1, d, padding_idx=0)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.05)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        genre_ids = torch.as_tensor(batch["genre_ids"], dtype=torch.long)
        genre_mask = torch.as_tensor(batch["genre_mask"], dtype=torch.long)
        genre_vec = self.genre_emb(genre_ids) * genre_mask.unsqueeze(-1).to(torch.float32)
        genre_vec = genre_vec.sum(1) / genre_mask.sum(1, keepdim=True).clamp(min=1).to(torch.float32)
        return {
            "user": self.user_emb(batch["user_idx"]),
            "item": self.item_emb(batch["item_idx"]),
            "gender": self.gender_emb(batch["gender_idx"]),
            "age": self.age_emb(batch["age_idx"]),
            "occ": self.occ_emb(batch["occ_idx"]),
            "genre": genre_vec,
        }


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, dropout: float) -> nn.Sequential:
    dims = [in_dim, *hidden, out_dim]
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------- DeepFM
class DeepFMRanker(nn.Module):
    """FM（一阶+二阶显式交叉）与 DNN（高阶隐式交叉）并联，共享 embedding。"""

    def __init__(self, cfg: RankConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = _SharedEmbedding(cfg)
        self.bias = nn.Parameter(torch.zeros(1))
        # 稠密特征的线性项（维度懒物化）
        self.dense_linear = nn.LazyLinear(1)
        d = cfg.id_dim
        self.user_dense_proj = nn.LazyLinear(16)
        self.item_dense_proj = nn.LazyLinear(16)
        dnn_in = 6 * d + 32
        self.dnn = _mlp(dnn_in, list(cfg.hidden), 1, cfg.dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.emb(batch)
        fields = [e["user"], e["item"], e["gender"], e["age"], e["occ"], e["genre"]]
        stack = torch.stack(fields, dim=1)                       # (B, F, K)
        # 一阶：bias + 各字段 embedding 逐维求和（等价于每个字段一个可学习线性打分）
        linear = self.bias + stack.sum(dim=(1, 2)) \
            + self.dense_linear(torch.cat([
                batch["user_dense"].to(torch.float32),
                batch["item_dense"].to(torch.float32)], dim=-1)).squeeze(-1)
        # 二阶：0.5 * [ (Σ_f v_f)^2 - Σ_f v_f^2 ] 逐维求和
        sum_v = stack.sum(dim=1)                                  # (B, K)
        square_sum = (sum_v * sum_v).sum(-1)
        sum_square = (stack * stack).sum(dim=(1, 2))
        pairwise = 0.5 * (square_sum - sum_square)
        # DNN 高阶交叉
        d = self.dnn(torch.cat([
            stack.flatten(1),
            self.user_dense_proj(batch["user_dense"].to(torch.float32)),
            self.item_dense_proj(batch["item_dense"].to(torch.float32)),
        ], dim=-1)).squeeze(-1)
        return torch.sigmoid(linear + pairwise + d)

    def n_params(self) -> int:
        # Lazy 模块（LazyLinear 等）在首次 forward 前参数是
        # UninitializedParameter，numel() 会报错——统计时跳过即可
        from torch.nn.parameter import UninitializedParameter
        return sum(p.numel() for p in self.parameters()
                   if not isinstance(p, UninitializedParameter))


# ------------------------------------------------------------------------ DIN
class AttentionPooling(nn.Module):
    """DIN 的 target attention：用候选物品当 query，对历史行为做加权池化。

    打分单元输入 [query, key, query-key, query*key]（外积式交互），
    经 MLP 输出每个历史位置的相关性分数，softmax 后加权求和。
    padding 位置分数置 -inf；全 padding 行退化为均匀分布（保证有限值）。
    """

    def __init__(self, query_dim: int, hidden: Sequence[int] = (64, 16)):
        super().__init__()
        dims = [4 * query_dim, *hidden, 1]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # query: (B, D)  keys: (B, L, D)  mask: (B, L)，1=真实行为
        L = keys.shape[1]
        q = query.to(keys.dtype).unsqueeze(1).expand(-1, L, -1)
        att_in = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        scores = self.mlp(att_in).squeeze(-1)                    # (B, L)
        keep = (mask > 0).to(torch.bool)
        scores = scores.masked_fill(~keep, float("-inf"))
        all_masked = ~keep.any(dim=1)
        scores = torch.where(
            all_masked.unsqueeze(1),
            torch.zeros_like(scores),                             # 全 padding 行 → 均匀权重
            scores,
        )
        w = torch.softmax(scores, dim=1)
        return (w.unsqueeze(-1) * keys).sum(dim=1)


class DINRanker(nn.Module):
    """DeepFM 基础上，把「用户历史」用 target attention 池化替代简单拼接。"""

    def __init__(self, cfg: RankConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = _SharedEmbedding(cfg)
        self.att = AttentionPooling(cfg.id_dim, (2 * cfg.id_dim, cfg.id_dim))
        self.user_dense_proj = nn.LazyLinear(16)
        self.item_dense_proj = nn.LazyLinear(16)
        d = cfg.id_dim
        dnn_in = 6 * d + d + 32                                 # 多一份 pooled 历史
        self.mlp = _mlp(dnn_in, list(cfg.hidden), 1, cfg.dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.emb(batch)
        hist = torch.as_tensor(batch["hist_item_idx"], dtype=torch.long)
        mask = (hist >= 0).to(torch.long)
        hist_idx = hist.clamp(min=0)
        keys = self.emb.item_emb(hist_idx)                        # (B, L, K)
        pooled = self.att(e["item"], keys, mask)                  # target attention
        x = torch.cat([
            e["user"], e["item"], e["gender"], e["age"], e["occ"], e["genre"], pooled,
            self.user_dense_proj(batch["user_dense"].to(torch.float32)),
            self.item_dense_proj(batch["item_dense"].to(torch.float32)),
        ], dim=-1)
        return torch.sigmoid(self.mlp(x).squeeze(-1))

    def n_params(self) -> int:
        # Lazy 模块（LazyLinear 等）在首次 forward 前参数是
        # UninitializedParameter，numel() 会报错——统计时跳过即可
        from torch.nn.parameter import UninitializedParameter
        return sum(p.numel() for p in self.parameters()
                   if not isinstance(p, UninitializedParameter))


# ----------------------------------------------------------------------- ESMM
class ESMM(nn.Module):
    """多目标全空间建模：CTR 塔与 CVR 塔共享底层 embedding。

    返回 (pctr, pcvr)；pCTCVR = pctr × pcvr 由调用方计算。
    CVR 的训练信号只来自「点击且转化」的样本，但通过 pCTCVR = pCTR×pCVR
    联合建模，让 CVR 塔在全曝光空间上更新，缓解 SSB（样本选择偏差）。
    """

    def __init__(self, cfg: RankConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = _SharedEmbedding(cfg)
        self.user_dense_proj = nn.LazyLinear(16)
        self.item_dense_proj = nn.LazyLinear(16)
        d = cfg.id_dim
        base = 6 * d + 32
        self.ctr_tower = _mlp(base, list(cfg.hidden), 1, cfg.dropout)
        self.cvr_tower = _mlp(base, list(cfg.hidden), 1, cfg.dropout)

    def _base(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.emb(batch)
        return torch.cat([
            e["user"], e["item"], e["gender"], e["age"], e["occ"], e["genre"],
            self.user_dense_proj(batch["user_dense"].to(torch.float32)),
            self.item_dense_proj(batch["item_dense"].to(torch.float32)),
        ], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._base(batch)
        pctr = torch.sigmoid(self.ctr_tower(x).squeeze(-1))
        pcvr = torch.sigmoid(self.cvr_tower(x).squeeze(-1))
        return pctr, pcvr

    def n_params(self) -> int:
        # Lazy 模块（LazyLinear 等）在首次 forward 前参数是
        # UninitializedParameter，numel() 会报错——统计时跳过即可
        from torch.nn.parameter import UninitializedParameter
        return sum(p.numel() for p in self.parameters()
                   if not isinstance(p, UninitializedParameter))


def esmm_loss(
    pctr: torch.Tensor, pcvr: torch.Tensor,
    y_click: torch.Tensor, y_convert: torch.Tensor,
) -> torch.Tensor:
    """ESMM 损失：BCE(pCTR, click) + BCE(pCTCVR, click×convert)。

    注意第二项用的是 pCTCVR = pCTR × pCVR —— 两个标签都定义在「全曝光空间」上，
    不存在「只在点击样本上训 CVR」的选择偏差。
    """
    y_click = y_click.to(torch.float32)
    y_convert = y_convert.to(torch.float32)
    pctcvr = pctr * pcvr
    loss_ctr = F.binary_cross_entropy(pctr.clamp(1e-6, 1 - 1e-6), y_click)
    loss_ctcvr = F.binary_cross_entropy(pctcvr.clamp(1e-6, 1 - 1e-6), y_click * y_convert)
    return loss_ctr + loss_ctcvr
