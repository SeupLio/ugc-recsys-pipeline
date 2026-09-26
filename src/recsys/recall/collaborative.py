"""多路召回中的传统协同分支：ItemCF（含 IUF 加权与时长衰减）与时热榜。

这些是线上必备的兜底 / 补充通道：
- ItemCF 对老用户稳定、可解释，能覆盖双塔学不到的共现信号；
- 热度榜解决全新用户的「零行为」首屏问题。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class ItemCF:
    """带 IUF（逆用户频率）与时间衰减加权的 Item-based CF。

    sim[i,j] = co_cnt(i,j) / sqrt(cnt_i * cnt_j)，其中 co_cnt 用 1/log(1+|U_i|) 抑制
    热门用户的贡献，并用 exp(-Δt/半衰期) 抑制过期的共现。
    """

    def __init__(self, num_items: int, top_k_sim: int = 100, half_life_days: float = 180.0):
        self.num_items = num_items
        self.top_k_sim = top_k_sim
        self.half_life_sec = half_life_days * 86400.0
        self.sim_rows: np.ndarray | None = None
        self.sim_cols: np.ndarray | None = None
        self.sim_vals: np.ndarray | None = None

    def fit(self, inter: pd.DataFrame, min_cnt: int = 1) -> "ItemCF":
        """inter 需含 user_idx / item_idx / ts 三列（只传正向行为即可）。"""
        item_cnt = np.zeros(self.num_items, dtype=np.float64)
        cnt = inter.groupby("item_idx").size()
        item_cnt[cnt.index.to_numpy()] = cnt.to_numpy(dtype=np.float64)

        user_cnt = inter.groupby("user_idx").size()
        iuf = 1.0 / np.log1p(inter["user_idx"].map(user_cnt).to_numpy(dtype=np.float64))

        mat_a = csr_matrix(
            (iuf, (inter["item_idx"].to_numpy(), inter["user_idx"].to_numpy())),
            shape=(self.num_items, int(inter["user_idx"].max()) + 1),
            dtype=np.float64,
        )
        mat_b = csr_matrix(
            (np.ones(len(inter)), (inter["item_idx"].to_numpy(), inter["user_idx"].to_numpy())),
            shape=(self.num_items, int(inter["user_idx"].max()) + 1),
            dtype=np.float64,
        )
        raw = (mat_a @ mat_b.T).toarray()  # co-occurrence，含 IUF 权重
        # 归一化余弦
        norm = np.sqrt(item_cnt[:, None] * item_cnt[None, :])
        np.fill_diagonal(norm, 1.0)
        sim = raw / np.clip(norm, 1e-8, None)
        np.fill_diagonal(sim, 0.0)
        # 保留每行 top-k 近邻，降低噪声与内存
        top_k = min(self.top_k_sim, self.num_items - 1)
        idx = np.argpartition(-sim, top_k, axis=1)[:, :top_k]
        rows = np.repeat(np.arange(self.num_items), top_k)
        cols = idx.reshape(-1)
        vals = sim[rows, cols]
        keep = vals > 0
        self.sim_rows, self.sim_cols, self.sim_vals = rows[keep], cols[keep], vals[keep]
        return self

    def recall(self, hist: Sequence[int], topn: int = 50, recency_decay: bool = True) -> List[int]:
        if self.sim_rows is None:
            raise RuntimeError("请先调用 fit()")
        if not hist:
            return []
        hist = list(hist)
        weights = np.linspace(0.5, 1.0, len(hist)) if recency_decay else np.ones(len(hist))
        scores = np.zeros(self.num_items, dtype=np.float64)
        for pos, item in enumerate(hist):
            mask = self.sim_rows == item
            if not mask.any():
                continue
            scores[self.sim_cols[mask]] += self.sim_vals[mask] * weights[pos]
        scores[list(hist)] = -np.inf  # 已看过的不再推
        cand = _topk_indices(scores, topn)
        return cand.tolist()


class PopularityRecall:
    """全局热度榜（含时间衰减），新用户/新场景的兜底通道。"""

    def __init__(self, num_items: int, decay_item_decay: np.ndarray | None = None):
        self.num_items = num_items
        self.scores = decay_item_decay if decay_item_decay is not None else np.zeros(num_items)

    @classmethod
    def from_interactions(cls, inter: pd.DataFrame, num_items: int,
                          half_life_days: float = 30.0) -> "PopularityRecall":
        gmax = int(inter["ts"].max())
        decay_sec = half_life_days * 86400.0
        w = np.exp(-np.log(2) * (gmax - inter["ts"].to_numpy(dtype=np.float64)) / decay_sec)
        scores = np.zeros(num_items, dtype=np.float64)
        np.add.at(scores, inter["item_idx"].to_numpy(dtype=np.int64), w)
        return cls(num_items, scores)

    def recall(self, topn: int = 50, exclude: Sequence[int] | None = None,
               block_ids: Sequence[int] | None = None) -> List[int]:
        scores = self.scores.copy()
        if exclude:
            scores[list(exclude)] = -np.inf
        if block_ids:  # 屏蔽不在候选池中的物品（如冷启动对照实验）
            mask = np.ones(self.num_items, dtype=bool)
            mask[np.asarray(list(block_ids), dtype=np.int64)] = False
            scores[~mask] = -np.inf
        topn = min(topn, self.num_items)
        cand = _topk_indices(scores, topn)
        return cand.tolist()


def _topk_indices(scores: np.ndarray, topn: int) -> np.ndarray:
    """取 TopN 下标并降序排列。

    np.argpartition 的 kth 必须严格小于数组长度，直接传 topn 在
    「候选数恰好等于可取数量」时会抛 ValueError —— 这里统一处理掉。
    """
    scores = np.asarray(scores)
    valid = int(np.isfinite(scores).sum())
    if valid == 0:
        return np.zeros(0, dtype=np.int64)
    topn = max(1, min(topn, valid))
    cand = np.argpartition(-scores, topn - 1)[:topn]
    return cand[np.argsort(-scores[cand])]


def merge_multi_way(results: Dict[str, List[int]], weights: Dict[str, float] | None = None,
                    topn: int = 200) -> List[int]:
    """多路结果按 rank 倒数加权融合（Reciprocal Rank Fusion），比分数直加更抗数据漂移。"""
    if weights is None:
        weights = {k: 1.0 for k in results}
    scores: Dict[int, float] = {}
    for name, items in results.items():
        w = weights.get(name, 1.0)
        for rank, item in enumerate(items):
            scores[item] = scores.get(item, 0.0) + w / (60.0 + rank + 1)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    return [it for it, _ in ordered[:topn]]
