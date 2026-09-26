"""离线评测指标：召回 / 排序 / 多目标 / 多样性。

所有函数均不依赖具体框架，输入为 Python 原生序列，便于单元测试直接构造真值校验。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np


# ------------------------------------------------------------------ 召回侧
def recall_at_k(ground_truth: Iterable[int], pred: Sequence[int], k: int) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    hit = len(set(list(pred)[:k]) & gt)
    return hit / len(gt)


def precision_at_k(ground_truth: Iterable[int], pred: Sequence[int], k: int) -> float:
    top = list(pred)[:k]
    if not top:
        return 0.0
    gt = set(ground_truth)
    return len(set(top) & gt) / len(top)


def hit_rate_at_k(ground_truth: Iterable[int], pred: Sequence[int], k: int) -> float:
    gt = set(ground_truth)
    return 1.0 if set(list(pred)[:k]) & gt else 0.0


def dcg(scores: Sequence[float]) -> float:
    return float(sum(s / np.log2(i + 2) for i, s in enumerate(scores)))


def ndcg_at_k(ground_truth: Iterable[int], pred: Sequence[int], k: int) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    top = list(pred)[:k]
    gains = [1.0 if it in gt else 0.0 for it in top]
    idcg = dcg([1.0] * min(len(gt), k))
    return dcg(gains) / idcg if idcg > 0 else 0.0


def average_precision_at_k(ground_truth: Iterable[int], pred: Sequence[int], k: int) -> float:
    gt = set(ground_truth)
    if not gt:
        return 0.0
    top = list(pred)[:k]
    hits, cum = 0, 0.0
    for i, it in enumerate(top):
        if it in gt:
            hits += 1
            cum += hits / (i + 1)
    return cum / min(len(gt), k)


def evaluate_topk(
    user2gt: Dict[int, List[int]], user2pred: Dict[int, List[int]], ks: Sequence[int] = (10, 20, 50)
) -> Dict[str, float]:
    """对整个测试集计算 Recall@{k}/NDCG@{k}/HR@{k}/MAP@{k}。"""
    out: Dict[str, float] = {}
    users = [u for u in user2gt if u in user2pred and user2gt[u]]
    for k in ks:
        rec = np.mean([recall_at_k(user2gt[u], user2pred[u], k) for u in users])
        nd = np.mean([ndcg_at_k(user2gt[u], user2pred[u], k) for u in users])
        hr = np.mean([hit_rate_at_k(user2gt[u], user2pred[u], k) for u in users])
        mp = np.mean([average_precision_at_k(user2gt[u], user2pred[u], k) for u in users])
        out[f"recall@{k}"] = float(rec)
        out[f"ndcg@{k}"] = float(nd)
        out[f"hit@{k}"] = float(hr)
        out[f"map@{k}"] = float(mp)
    out["users"] = len(users)
    return out


# ------------------------------------------------------------------ 排序侧
def auc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    """基于 rank 的 AUC，等价于 ROC 曲线下面积（含并列修正）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based 平均秩
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gauc(
    y_true: Sequence[float],
    y_score: Sequence[float],
    group_ids: Sequence[int | str],
    min_group_size: int = 3,
) -> float:
    """GAUC：按用户分组算 AUC 后加权平均。

    组内只有单一类别的用户不计入（AUC 无定义），避免把 1.0/0.0 噪声灌入均值。
    """
    buckets: Dict[Any, List[int]] = defaultdict(list)
    for i, g in enumerate(group_ids):
        buckets[g].append(i)
    num, den = 0.0, 0.0
    skipped = 0
    for idx in buckets.values():
        if len(idx) < min_group_size:
            skipped += 1
            continue
        ys = [y_true[i] for i in idx]
        if len(set(ys)) < 2:
            skipped += 1
            continue
        a = auc(ys, [y_score[i] for i in idx])
        if a != a:  # NaN
            skipped += 1
            continue
        num += a * len(idx)
        den += len(idx)
    return float(num / den) if den > 0 else float("nan")


def log_loss(y_true: Sequence[float], y_score: Sequence[float], eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_score, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman 相关系数（用秩实现，避免依赖 scipy 版本行为差异）。"""
    rx = _rank(x)
    ry = _rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / denom) if denom > 0 else float("nan")


def _rank(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    # 处理并列取平均秩
    sorted_arr = arr[order]
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def lift_at_k(y_true: Sequence[float], y_score: Sequence[float], top_ratio: float = 0.1) -> float:
    """Top-k 命中率 / 全量命中率，用于人群包、LTV 等增长场景评估。"""
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    base = y.mean()
    if base == 0:
        return float("nan")
    k = max(1, int(len(y) * top_ratio))
    top_idx = np.argsort(-s)[:k]
    return float(y[top_idx].mean() / base)


# ------------------------------------------------------------------ 生态侧
def coverage(pred_lists: Iterable[Sequence[int]], num_items: int) -> float:
    seen = set()
    for lst in pred_lists:
        seen.update(lst)
    return len(seen) / num_items if num_items else 0.0


def intra_list_diversity(pred: Sequence[int], item_feat: Dict[int, Sequence[int]]) -> float:
    """候选列表内部两两 Jaccard 距离的均值：越大说明推荐越不同质。"""
    items = [it for it in pred if it in item_feat]
    if len(items) < 2:
        return 0.0
    feats = [set(item_feat[it]) for it in items]
    tot, cnt = 0.0, 0
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            union = len(feats[i] | feats[j])
            inter = len(feats[i] & feats[j])
            tot += 1.0 - (inter / union if union else 1.0)
            cnt += 1
    return tot / cnt if cnt else 0.0


def novelty(pred_lists: Iterable[Sequence[int]], item_pop: Dict[int, int]) -> float:
    """平均流行度负对数：值越大表示越少推荐头部内容。"""
    vals = []
    for lst in pred_lists:
        for it in lst:
            if it in item_pop:
                vals.append(np.log2(1.0 + item_pop[it]))
    return float(np.mean(vals)) if vals else 0.0
