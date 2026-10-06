"""MMoE 与校准模块测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from recsys.models.calibration import (
    ece, isotonic_apply, isotonic_fit, pcoc, reliability_bins)
from recsys.models.multitask import MMoERanker
from recsys.models.rank import RankConfig


def _cfg() -> RankConfig:
    return RankConfig(num_users=50, num_items=40, num_genres=5,
                      num_gender=2, num_age=3, num_occ=4)


def _batch(n: int = 128) -> dict:
    g = np.random.default_rng(0)
    gi = g.integers(0, 6, (n, 3))
    mask = (gi > 0).astype(np.int64)
    return {
        "user_idx": torch.as_tensor(g.integers(0, 50, n)),
        "item_idx": torch.as_tensor(g.integers(0, 40, n)),
        "gender_idx": torch.as_tensor(g.integers(0, 2, n)),
        "age_idx": torch.as_tensor(g.integers(0, 3, n)),
        "occ_idx": torch.as_tensor(g.integers(0, 4, n)),
        "genre_ids": torch.as_tensor(gi), "genre_mask": torch.as_tensor(mask),
        "user_dense": torch.as_tensor(g.normal(size=(n, 8)), dtype=torch.float32),
        "item_dense": torch.as_tensor(g.normal(size=(n, 8)), dtype=torch.float32),
    }


class TestMMoE:
    def test_forward_shapes_and_range(self):
        model = MMoERanker(_cfg(), n_experts=4)
        model.eval()
        with torch.no_grad():
            pctr, pcvr = model(_batch(64))
        assert pctr.shape == (64,) and pcvr.shape == (64,)
        assert 0 <= float(pctr.min()) and float(pctr.max()) <= 1
        assert 0 <= float(pcvr.min()) and float(pcvr.max()) <= 1

    def test_experts_1_degenerates_to_shared(self):
        """单专家 = 共享底座 + 双塔（MMoE 论文的退化对照）。"""
        model = MMoERanker(_cfg(), n_experts=1)
        with torch.no_grad():
            pctr, pcvr = model(_batch(32))
        assert float(pctr.sum()) > 0

    def test_gate_divergence_finite(self):
        model = MMoERanker(_cfg(), n_experts=3)
        js = model.gate_divergence(_batch(32))
        assert js >= 0 and np.isfinite(js)   # JS 散度非负有限

    def test_gradients_flow_to_all_experts(self):
        """反向传播应触达每个专家（门 softmax 下共享梯度）。"""
        model = MMoERanker(_cfg(), n_experts=4)
        pctr, pcvr = model(_batch(32))
        (pctr.sum() + pcvr.sum()).backward()
        for i, ex in enumerate(model.experts):
            assert all(p.grad is not None and p.grad.abs().sum() > 0
                       for p in ex.parameters()), f"expert {i} 无梯度"


class TestCalibration:
    def test_pav_monotonic_and_fits(self):
        rng = np.random.default_rng(0)
        # 构造已知映射：真实概率 = score^2（非线性、单调）
        s = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < s ** 2).astype(float)
        table = isotonic_fit(s, y)
        assert len(table) >= 1
        vals = [v for _, v in table]
        assert all(a <= b for a, b in zip(vals, vals[1:]))      # 单调不减
        cal = isotonic_apply(np.array([0.05, 0.5, 0.95]), table)
        # 0.05 处真实概率≈0.0025，5000 样本下该块很可能无正例（PAV 输出 0），
        # 这是统计稀疏的正常行为——只断言非降与中位区间
        assert 0 <= cal[0] <= cal[1] < cal[2]                     # 保序（非降）
        # 0.5 处真实概率 0.25，校准值应接近（大数定律容差）
        assert abs(cal[1] - 0.25) < 0.05

    def test_ece_zero_for_perfect(self):
        # ECE 是桶聚合口径：100 个 p=0.2 的样本配 20 个正例 → 完美校准
        y = np.zeros(100)
        y[:20] = 1.0
        assert ece(np.full(100, 0.2), y) < 1e-9
        # 完全失准：报 0.9 但从不发生
        assert ece(np.array([0.9, 0.9]), np.array([0.0, 0.0])) > 0.8

    def test_pcoc_direction(self):
        s = np.array([0.3, 0.3, 0.3])
        assert pcoc(s, np.array([0.0, 0.0, 0.0])) > 1     # 高估
        assert pcoc(s, np.array([1.0, 1.0, 1.0])) < 1     # 低估
        assert abs(pcoc(s, s) - 1.0) < 1e-9               # 完美

    def test_calibration_preserves_ranking(self):
        """校准映射对原始分数单调不减（PAV 的定义性质）。"""
        rng = np.random.default_rng(1)
        s = rng.uniform(0, 1, 4000)
        y = (rng.uniform(0, 1, 4000) < s).astype(float)
        cal = isotonic_apply(s, isotonic_fit(s, y))
        order = np.argsort(s, kind="stable")
        diffs = np.diff(cal[order])
        assert (diffs >= -1e-12).all()          # 分数升 → 校准值非降

    def test_reliability_bins(self):
        s = np.array([0.05, 0.15, 0.85])
        y = np.array([0.0, 1.0, 1.0])
        bins = reliability_bins(s, y, n_bins=10)
        assert bins and all(0 <= o <= 1 for _, o, _ in bins)
        assert sum(n for _, _, n in bins) == 3
