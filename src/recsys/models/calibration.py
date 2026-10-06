"""pCTR 概率校准（Probability Calibration）。

为什么校准是推荐/广告的「隐性刚需」：
- AUC 只看序（排序质量），计费 / 竞价 / EE 探索 / 分桶流量预估看的是
  「绝对值」——pCTR=0.12 意味着每千次曝光 120 次点击，模型分数整体
  偏高 2 倍就是真金白银的损失；
- 神经网络 BCE 训练在类别不平衡 / 负采样校准（in-batch negative 的
  logQ 修正）下，分数分布天然不等于真实概率。

实现（零第三方依赖）：
- isotonic_fit：PAV（Pool Adjacent Violators）保序回归，输出分段
  线性单调映射；
- isotonic_apply：按阈值二分查表外推；
- ece：Expected Calibration Error（15 桶，概率 vs 实测频率）；
- pcoc：Predicted-over-Observed Clicks（整体高估/低估倍数）。

协议：校准器只在 valid 的一半样本上拟合，另一半上报 ECE——
「用测试集自己拟合校准器」是常见的数据泄漏，这里防住了。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------- PAV 保序回归 ----------------

def isotonic_fit(scores: np.ndarray, labels: np.ndarray) -> List[Tuple[float, float]]:
    """PAV 算法：返回 [(阈值下界, 校准值), ...] 的单调阶梯。

    对 (score, label) 按 score 排序，迭代合并「违反单调」的相邻块
    （后块均值 < 前块均值则合并），直到整体单调不减。
    """
    assert len(scores) == len(labels) and len(scores) > 0
    order = np.argsort(scores, kind="stable")
    y = labels[order].astype(np.float64)
    x = scores[order].astype(np.float64)

    blocks = []  # 每个 block: [y_sum, n, x_start, x_end]
    for i in range(len(y)):
        blocks.append([y[i], 1, x[i], x[i]])
        # 后块均值低于前块 → 合并（消除违反单调性），循环直到稳定
        while len(blocks) >= 2 and \
                blocks[-1][0] / blocks[-1][1] < blocks[-2][0] / blocks[-2][1]:
            a, b = blocks[-2], blocks[-1]
            blocks[-2] = [a[0] + b[0], a[1] + b[1], a[2], b[3]]
            blocks.pop()
    return [(b[2], b[0] / b[1]) for b in blocks]


def isotonic_apply(scores: np.ndarray, table: List[Tuple[float, float]]) -> np.ndarray:
    """按查表阈值二分定位（阶梯外延：低于首阈值用首值，高于末阈值用末值）。"""
    if len(scores) == 0:
        return np.empty(0)
    th = np.array([t[0] for t in table])
    val = np.array([t[1] for t in table])
    idx = np.searchsorted(th, scores, side="right") - 1
    idx = np.clip(idx, 0, len(val) - 1)
    return val[idx]


# ---------------- 校准诊断指标 ----------------

def ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error：Σ |桶频 - 桶均分| × 桶样本占比。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += m.sum() / len(scores) * abs(labels[m].mean() - scores[m].mean())
    return float(total)


def pcoc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Predicted-over-Observed：>1 系统性高估，<1 低估（计费口径的生命线指标）。"""
    labels = np.asarray(labels, dtype=np.float64)
    return float(scores.sum() / max(labels.sum(), 1e-12))


def reliability_bins(scores: np.ndarray, labels: np.ndarray,
                     n_bins: int = 10) -> List[Tuple[float, float, int]]:
    """可靠性曲线数据：[(桶中心, 实测 CTR, 样本数), ...]。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append(((edges[b] + edges[b + 1]) / 2, float(labels[m].mean()), int(m.sum())))
    return out
