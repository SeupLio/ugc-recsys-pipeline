"""离线指标单测：手写实现与权威实现对拍，并对着真值算一遍。"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from recsys.eval import metrics as M


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, size=200).astype(float)
        s = rng.normal(size=200)
        assert np.isclose(M.auc(y, s), roc_auc_score(y, s), atol=1e-10)


def test_auc_with_ties():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    s = np.array([0.5, 0.5, 0.2, 0.9])
    # 并列取平均秩：正样本秩 2.5 与 4，负样本秩 2.5 与 1
    # AUC = (6.5 - 2*3/2) / (2*2) = 0.875
    assert np.isclose(M.auc(y, s), 0.875)
    assert np.isclose(M.auc(y, s), roc_auc_score(y, s), atol=1e-10)


def test_auc_degenerate():
    assert np.isnan(M.auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_topk_metrics_against_hand_computation():
    gt = [1, 2, 3]
    pred = [9, 3, 1, 7, 2]
    assert np.isclose(M.recall_at_k(gt, pred, 5), 1.0)
    assert np.isclose(M.recall_at_k(gt, pred, 2), 1 / 3)
    assert np.isclose(M.precision_at_k(gt, pred, 5), 3 / 5)
    assert np.isclose(M.hit_rate_at_k(gt, pred, 2), 1.0)
    # NDCG@3：预测列表 [9,3,1,7,2]，前 3 位命中 3(idx1) 与 1(idx2)
    # DCG = 0 + 1/log2(3) + 1/log2(4)；IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
    dcg = 1 / np.log2(3) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
    assert np.isclose(M.ndcg_at_k(gt, pred, 3), dcg / idcg, atol=1e-9)
    assert M.ndcg_at_k(gt, pred, 3) < 1.0, "顺序未完全对齐时应小于 1"


def test_empty_ground_truth_is_zero():
    assert M.recall_at_k([], [1, 2, 3], 5) == 0.0
    assert M.ndcg_at_k([], [1, 2, 3], 5) == 0.0


def test_gauc_weights_by_group_size():
    # 用户 A：完美排序(AUC=1)，用户 B：反序(AUC=0)，样本量 2:2 → GAUC=0.5
    y = [0.0, 1.0, 0.0, 1.0]
    s = [0.1, 0.9, 0.9, 0.1]
    g = ["A", "A", "B", "B"]
    assert np.isclose(M.gauc(y, s, g, min_group_size=1), 0.5)


def test_gauc_skips_single_class_group():
    y = [0.0, 0.0, 1.0, 0.0]
    s = [0.1, 0.2, 0.9, 0.3]
    g = ["A", "A", "B", "B"]  # A 全负样本，应被跳过
    val = M.gauc(y, s, g, min_group_size=1)
    assert not np.isnan(val)
    assert 0 <= val <= 1


def test_coverage_and_diversity():
    preds = [[1, 2], [2, 3], [1]]
    assert np.isclose(M.coverage(preds, 4), 3 / 4)
    feat = {1: [0, 1], 2: [0, 1], 3: [2]}
    # 列表 [1,2] 内部完全同质 → 0；加入 3 后距离上升
    assert M.intra_list_diversity([1, 2], feat) == 0.0
    assert M.intra_list_diversity([1, 3], feat) > 0.9


def test_lift_positive():
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 9.9])
    assert np.isclose(M.lift_at_k(y, s, 0.1), 10.0)


def test_spearman():
    x = [1, 2, 3, 4, 5]
    assert np.isclose(M.spearman(x, x), 1.0)
    assert np.isclose(M.spearman(x, [5, 4, 3, 2, 1]), -1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
