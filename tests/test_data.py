"""数据侧单测：护栏性质。

这一类测试不是为了「覆盖率」，而是防止最致命的两类线上事故：
1) 训练集与测试集穿越 → 离线指标虚高、线上崩盘；
2) 负采样把用户看过的东西当负例 → 模型学到错误的负信号。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from recsys.data import builder
from recsys.data.torch_data import causal_truncate

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_pad_seq_right_aligns():
    arr = builder.pad_seq([1, 2, 3], max_len=5)
    assert arr.tolist() == [-1, -1, 1, 2, 3]


def test_pad_seq_truncates_to_recent():
    arr = builder.pad_seq(list(range(10)), max_len=3)
    assert arr.tolist() == [7, 8, 9]


def test_pad_genres_mask():
    ids, mask = builder.pad_genres([3, 5], max_genres=4)
    assert ids.tolist() == [3, 5, 0, 0]
    assert mask.tolist() == [1, 1, 0, 0]


def test_negative_sampler_avoids_seen():
    counts = np.array([10.0, 5.0, 1.0, 1.0, 100.0])
    sampler = builder.NegativeSampler(5, counts, beta=0.75, seed=0)
    seen = {0, 1, 2, 3}
    for _ in range(50):
        got = sampler.sample(seen, 3)
        assert len(got) == 3
        assert all(g not in seen for g in got)


def test_negative_sampler_popularity_skew():
    num_items = 4
    counts = np.array([1000.0, 1.0, 1.0, 1.0])
    sampler = builder.NegativeSampler(num_items, counts, beta=0.0, seed=1)  # beta=0 → 近似均匀
    hit = 0
    n = 4000
    draws = sampler.sample_batch(np.zeros(n, dtype=int), {0: set()})
    hit = int((draws == 0).sum())
    assert hit / n < 0.4  # beta=0 时应接近 25%，显著低于 1000/1003


def test_causal_truncate_keeps_prefix():
    import torch

    hist = torch.tensor([[-1, -1, 7, 8, 9], [-1, 5, 6, 7, 8]])
    keep = np.array([2, 0])
    out = causal_truncate(hist, keep)
    assert out[0].tolist() == [-1, -1, 7, 8, -1]  # 只留前 2 条
    assert out[1].tolist() == [-1, -1, -1, -1, -1]  # keep=0 → 全屏蔽


def test_causal_truncate_removes_target_item():
    """目标物品必须被移出历史，否则 DIN 会学到 '历史里有→点击' 的伪规律。"""
    import torch

    hist = torch.tensor([[-1, 3, 7, 8, 9]])
    keep = np.array([3])  # 窗口内有 4 条行为，目标位于最后一条 → 只保留它之前的 3 条
    out = causal_truncate(hist, keep)
    assert int(out[0][-1]) == -1, "曝光时刻之后的行为必须被屏蔽"
    assert out[0].tolist() == [-1, 3, 7, 8, -1]


@pytest.mark.skipif(not (PROC / "rank_hist_keep.npy").exists(), reason="需要先跑 01_prepare_data.py")
def test_processed_dataset_has_no_train_test_leak():
    import pandas as pd

    inter = pd.read_csv(PROC / "interactions.csv")
    train = inter[inter["split_tag"] == "train"]
    test = inter[inter["split_tag"] == "test"]
    overlap = set(zip(train["user_idx"], train["item_idx"])) & set(
        zip(test["user_idx"], test["item_idx"])
    )
    assert not overlap, "测试集与训练集存在完全相同的 (user, item) 记录"


@pytest.mark.skipif(not (PROC / "rank_hist_keep.npy").exists(), reason="需要先跑 01_prepare_data.py")
def test_causal_mask_file_consistency():
    users = np.load(PROC / "rank_users.npy")
    keep = np.load(PROC / "rank_hist_keep.npy")
    assert len(users) == len(keep)
    # 同一用户同一批样本（正样本 1 条 + 4 条负样本）必须共享同一个 keep 值，
    # 否则模型能从历史长度反推出标签。
    assert len(users) % 5 == 0
    blocks = keep.reshape(-1, 5)
    assert np.all(blocks == blocks[:, [0]])


@pytest.mark.skipif(not (PROC / "cold_items.npy").exists(), reason="需要先跑 01_prepare_data.py")
def test_cold_items_excluded_from_training():
    import pandas as pd

    cold = np.load(PROC / "cold_items.npy")
    inter = pd.read_csv(PROC / "interactions.csv")
    train_pairs = inter[inter["split_tag"].isin(["train", "valid"])]
    leaked = set(cold.tolist()) & set(train_pairs["item_idx"].unique().tolist())
    assert not leaked, "冷启动物品出现在了训练/验证集中"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
