"""MMoE 多任务精排（Multi-gate Mixture-of-Experts, Ma et al. KDD'18）。

与 ESMM 的结构性对照（同一份输入、同一套标签、同一个损失）：

- ESMM：共享子网络「串联」两个任务——CTR 与 CVR 共用全部底层表达，
  任务间互相牵制（跷跷板现象）；
- MMoE：N 个专家并行，**每个任务一个独立 softmax 门**学自己的专家
  权重——任务相关时门趋同（近似共享），任务冲突时门分化（自动解耦），
  是工业界多目标精排（→ PLE 的前置基础）的标准起点。

输出 (pCTR, pCVR)，pCTCVR = pCTR × pCVR；损失沿用 esmm_loss
（两个标签都定义在全曝光空间，无 CVR 选择偏差）。
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rank import RankConfig, _SharedEmbedding, _mlp


class MMoERanker(nn.Module):
    """共享 embedding → 并行专家 × 任务门 → 任务塔 → pCTR / pCVR。"""

    def __init__(self, cfg: RankConfig, n_experts: int = 4,
                 expert_hidden: Sequence[int] = (64, 32),
                 expert_out: int = 16, tower_hidden: Sequence[int] = (32,)) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = _SharedEmbedding(cfg)
        d = cfg.id_dim
        # 稠密投影与 DeepFM 同构（LazyLinear 懒物化输入维度）
        self.user_dense_proj = nn.LazyLinear(16)
        self.item_dense_proj = nn.LazyLinear(16)
        in_dim = 6 * d + 32

        self.n_experts = int(n_experts)
        self.experts = nn.ModuleList(
            [_mlp(in_dim, list(expert_hidden), expert_out, cfg.dropout)
             for _ in range(n_experts)])
        # 任务门：输入同底座特征，输出对 n 个专家的 softmax 权重
        self.gate_ctr = _mlp(in_dim, [32], n_experts, 0.0)
        self.gate_cvr = _mlp(in_dim, [32], n_experts, 0.0)
        self.tower_ctr = _mlp(expert_out, list(tower_hidden), 1, cfg.dropout)
        self.tower_cvr = _mlp(expert_out, list(tower_hidden), 1, cfg.dropout)

    def _base(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.emb(batch)
        stack = torch.stack([e["user"], e["item"], e["gender"],
                             e["age"], e["occ"], e["genre"]], dim=1)
        return torch.cat([
            stack.flatten(1),
            self.user_dense_proj(batch["user_dense"].to(torch.float32)),
            self.item_dense_proj(batch["item_dense"].to(torch.float32)),
        ], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]):
        x = self._base(batch)
        exp_out = torch.stack([m(x) for m in self.experts], dim=1)   # (B, E, H)
        g_ctr = F.softmax(self.gate_ctr(x), dim=-1)                  # (B, E)
        g_cvr = F.softmax(self.gate_cvr(x), dim=-1)
        c_ctr = (exp_out * g_ctr.unsqueeze(-1)).sum(1)               # (B, H)
        c_cvr = (exp_out * g_cvr.unsqueeze(-1)).sum(1)
        pctr = torch.sigmoid(self.tower_ctr(c_ctr)).squeeze(-1)
        pcvr = torch.sigmoid(self.tower_cvr(c_cvr)).squeeze(-1)
        return pctr, pcvr

    def gate_divergence(self, batch: Dict[str, torch.Tensor]) -> float:
        """两任务门权重的平均 JS 散度（>0 说明学到了任务分化）。

        面试可讲：观察门分化程度 = 多任务是否真的在「各取所需」。
        """
        x = self._base(batch)
        g1 = F.softmax(self.gate_ctr(x), dim=-1)
        g2 = F.softmax(self.gate_cvr(x), dim=-1)
        m = 0.5 * (g1 + g2)
        js = 0.5 * (F.kl_div(m.log(), g1, reduction="batchmean", log_target=False)
                    + F.kl_div(m.log(), g2, reduction="batchmean", log_target=False))
        return float(js.detach())

    def n_params(self) -> int:
        from torch.nn.parameter import UninitializedParameter
        return sum(p.numel() for p in self.parameters()
                   if not isinstance(p, UninitializedParameter))
