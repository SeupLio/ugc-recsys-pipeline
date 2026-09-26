"""新增能力（内容质量/意图/难负例/在线模拟）的行为测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _import(name):
    import importlib
    return importlib.import_module(name)


# ------------------------------------------------------------------ 质量评估
def test_bayesian_smoothing_shrinks_low_count():
    q = _import("11_content_quality")
    # 1 次曝光 1 次点击：原始 CTR=1.0，平滑后应大幅向先验收缩
    smoothed = q.bayesian_smooth_rate(np.array([1.0]), np.array([1.0]), prior=0.05, m=50.0)
    assert smoothed[0] < 0.1, "低曝光的高 CTR 必须被显著收缩"
    # 高曝光时平滑后应接近真实比率
    big = q.bayesian_smooth_rate(np.array([900.0]), np.array([1000.0]), prior=0.05, m=50.0)
    assert 0.8 < big[0] < 0.95


def test_gini_range():
    q = _import("11_content_quality")
    assert abs(q.gini(np.ones(100))) < 1e-6          # 完全均匀 -> 0
    concentrated = np.zeros(100); concentrated[0] = 1.0
    assert q.gini(concentrated) > 0.9                 # 完全集中 -> ~1


# ------------------------------------------------------------------ 在线模拟
def test_srm_check_flags_imbalance():
    sim = _import("14_online_sim")
    ok = sim.srm_check(5000, 5000)
    bad = sim.srm_check(5000, 4000)
    assert ok["pass"] is True
    assert bad["pass"] is False


def test_cuped_reduces_variance_when_correlated():
    sim = _import("14_online_sim")
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = 0.8 * x + rng.normal(size=5000) * 0.3  # 与 x 强相关
    adj = sim.cuped_adjust(y, x)
    assert adj.var() < y.var(), "CUPED 应削减方差"


def test_cuped_no_effect_when_uncorrelated():
    sim = _import("14_online_sim")
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    y = rng.normal(size=5000)  # 与 x 无关
    adj = sim.cuped_adjust(y, x)
    # 不应显著改变方差（允许小幅波动）
    assert abs(adj.var() - y.var()) < 0.05 * y.var()


def test_mde_decreases_with_sample_size():
    sim = _import("14_online_sim")
    small = sim.mde_required(0.5, 1000)
    large = sim.mde_required(0.5, 100000)
    assert large < small, "样本越大 MDE 越小"
    assert small > 0


def test_bootstrap_ratio_ci_contains_point_estimate():
    sim = _import("14_online_sim")
    rng = np.random.default_rng(2)
    num = rng.integers(0, 10, size=2000).astype(float)
    den = np.full(2000, 10.0)
    unit = rng.integers(0, 500, size=2000)
    R, lo, hi = sim.bootstrap_ratio_ci(num, den, unit)
    assert lo <= R <= hi


def test_solve_intercept_shift_hits_target():
    sim = _import("14_online_sim")
    z = np.random.default_rng(3).normal(size=5000)
    delta = sim._solve_intercept_shift(z, 0.3)
    achieved = np.mean(1.0 / (1.0 + np.exp(-(z + delta))))
    assert abs(achieved - 0.3) < 0.01, "截距校准后均值应命中目标 CTR"


def test_allocate_respects_cold_quota():
    sim = _import("14_online_sim")
    rng = np.random.default_rng(4)
    scores = rng.normal(size=200)
    cold_mask = np.zeros(200, dtype=bool)
    cold_mask[[3, 10, 50, 120]] = True  # 4 个新内容
    chosen = sim.allocate(scores, cold_mask, quota=2, epsilon=0.0, rng=rng)
    cold_chosen = cold_mask[chosen].sum()
    assert cold_chosen == 2, "应按额度精确保留新内容坑位"
    assert len(chosen) == sim.POOL


# ------------------------------------------------------------------ 难负例
def test_causal_keep_monotonic_within_user():
    """同一用户的 keep 值应随行为序单调不减，且每条正样本的负例块共享同值。"""
    import pandas as pd
    hn = _import("12_hard_negative")
    pos = pd.DataFrame({
        "user_idx": [0, 0, 0, 1, 1],
        "item_idx": [5, 6, 7, 8, 9],
        "ts": [1, 2, 3, 1, 2],
    })
    keep = hn._causal_keep(pos, stride=5)
    assert len(keep) == 25  # 5 正样本 × (1 正 + 4 负)
    # 每个 stride 块内 keep 相同
    assert keep[0] == keep[1] == keep[4]
    # 用户内单调不减
    u0 = keep[0::5][:3]
    assert list(u0) == sorted(u0)


def test_hard_eval_excludes_positive_item():
    """难负例评测集里，正样本物品不能出现在它自己的负例中。"""
    # 直接构造 hard 字典验证 make_hard_eval_set 的过滤
    hn = _import("12_hard_negative")

    class _Data:
        interactions = None

    # 用 monkeypatch 方式构造最小 data
    import pandas as pd
    data = _Data()
    data.interactions = pd.DataFrame({
        "user_idx": [0, 0], "item_idx": [5, 5], "rating": [4, 5],
        "split_tag": ["valid", "valid"], "ts": [1, 2],
    })
    hard = {0: np.array([5, 6, 7, 8, 9, 10])}
    u, i, y = hn.make_hard_eval_set(data, hard, seed=0, n_neg=4)
    # 正样本 (user0, item5) 的负例里不应包含 item5
    pos_rows = np.where(y == 1)[0]
    for pr in pos_rows:
        neg_block = i[pr + 1: pr + 5]
        assert 5 not in neg_block


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
