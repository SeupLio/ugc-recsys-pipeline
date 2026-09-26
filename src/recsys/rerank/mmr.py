"""重排：MMR 多样性打散 + 类目约束，解决精排结果同质化。

精排逐条打分不考虑结果之间的相互影响，容易推荐一整屏同类内容。
MMR 在「相关性」与「多样性」之间做显式权衡。
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def mmr_rerank(
    candidates: Sequence[int],
    scores: Dict[int, float],
    item_sim_fn,
    topn: int = 10,
    lambda_div: float = 0.5,
) -> list[int]:
    """贪心 MMR：每步选择 λ·相关性 - (1-λ)·与已选最大相似度 最大的候选。"""
    remain = list(candidates)
    selected: list[int] = []
    while remain and len(selected) < topn:
        best, best_val = None, -np.inf
        for c in remain:
            rel = scores.get(c, 0.0)
            div = max((item_sim_fn(c, s) for s in selected), default=0.0)
            val = lambda_div * rel - (1.0 - lambda_div) * div
            if val > best_val:
                best, best_val = c, val
        selected.append(best)
        remain.remove(best)
    return selected


def jaccard_sim_fn(item_feat: Dict[int, Sequence[int]]):
    """基于类目集合的 Jaccard 相似度，用于多样性惩罚项。"""

    def sim(a: int, b: int) -> float:
        fa, fb = set(item_feat.get(a, [])), set(item_feat.get(b, []))
        union = fa | fb
        return len(fa & fb) / len(union) if union else 0.0

    return sim


def cosine_sim_fn(vectors: np.ndarray):
    """基于内容向量的余弦相似度（已归一化时直接点积）。"""
    v = np.asarray(vectors, dtype=np.float32)

    def sim(a: int, b: int) -> float:
        return float(np.dot(v[a], v[b]))

    return sim


def category_cap(
    candidates: Sequence[int],
    item_feat: Dict[int, Sequence[int]],
    topn: int = 10,
    max_per_cat: int = 3,
) -> list[int]:
    """硬约束：同一一级类目最多出现 max_per_cat 条，保证首屏类目覆盖。"""
    out: list[int] = []
    cnt: Dict[int, int] = {}
    spill: list[int] = []
    for c in candidates:
        cats = set(item_feat.get(c, []))
        if not cats:
            out.append(c)
            continue
        peak = max(cnt.get(cat, 0) for cat in cats)
        if peak < max_per_cat and len(out) < topn:
            out.append(c)
            for cat in cats:
                cnt[cat] = cnt.get(cat, 0) + 1
        else:
            spill.append(c)
    while len(out) < topn and spill:
        out.append(spill.pop(0))
    return out[:topn]
