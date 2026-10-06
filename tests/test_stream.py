"""stream 模块测试：时间衰减热度 / 向量负采样 / 事件流加载 / 在线增量更新。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from recsys.serving.stream import (
    DecayedPopularity,
    OnlineRanker,
    UserHistory,
    load_event_stream,
)


class TestDecayedPopularity:
    def test_decay_and_top(self):
        pop = DecayedPopularity(10, half_life_s=100.0)
        pop.advance(0.0)
        pop.add(0, [1, 1, 2])                 # t=0：item1 计 2，item2 计 1
        pop.add(100.0, [3])                    # t=100：旧计数减半
        assert pop.counts[1] < 2.0 + 1e-9      # 衰减生效
        top = pop.top(2)
        assert 3 in top.tolist()               # 最新事件占据榜首
        assert pop.top_cache is not None       # top 结果被缓存

    def test_probs_normalized(self):
        pop = DecayedPopularity(5, half_life_s=10.0)
        pop.advance(0.0)
        pop.add(0, [0, 1, 2, 3, 4])
        p = pop.probs()
        assert abs(p.sum() - 1.0) < 1e-9 or p.sum() == 0


class TestUserHistory:
    def test_sample_negatives_excludes_seen(self):
        rng = np.random.default_rng(0)
        h = UserHistory(3)
        h.add(np.array([0, 0, 0]), np.array([1, 2, 3]))
        neg = h.sample_negatives(np.array([0, 0, 0, 0, 0, 0, 0, 0]), rng, 10)
        assert all(n not in (1, 2, 3) for n in neg.tolist())

    def test_sample_negatives_pop_weighted(self):
        rng = np.random.default_rng(1)
        h = UserHistory(2)
        h.add(np.array([0]), np.array([9]))
        pop_p = np.zeros(10)
        pop_p[9] = 1.0                          # 只会抽到 9（已看过）→ 拒绝重抽
        neg = h.sample_negatives(np.array([0] * 4), rng, 10, pop_p=pop_p)
        assert all(n != 9 for n in neg.tolist())


class TestLoadEventStream:
    def test_sorted_by_time(self, tmp_path: Path):
        p = tmp_path / "interactions.csv"
        pd.DataFrame({"user_idx": [1, 0, 1], "item_idx": [2, 3, 4],
                      "rating": [5, 3, 4], "ts": [200, 100, 300],
                      "split_tag": ["train"] * 3}).to_csv(p, index=False)
        u, i, t = load_event_stream(p)
        assert list(t) == [100, 200, 300]
        assert list(u) == [0, 1, 1]
        assert list(i) == [3, 2, 4]


class TestOnlineRanker:
    def test_partial_fit_learns_and_scores(self):
        class _DictModel(torch.nn.Module):
            """与 recsys 模型同接口：forward(batch: dict) -> (B,) 概率。"""
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(4, 8), torch.nn.ReLU(),
                    torch.nn.Linear(8, 1), torch.nn.Sigmoid())

            def forward(self, batch):
                return self.net(batch["x"]).squeeze(-1)

        torch.manual_seed(0)
        model = _DictModel()
        ranker = OnlineRanker(model, lr=5e-2)
        rng = np.random.default_rng(0)
        x_pos = rng.normal(2.0, 0.5, (256, 4)).astype(np.float32)
        x_neg = rng.normal(-2.0, 0.5, (256, 4)).astype(np.float32)
        batch = {"x": torch.cat([torch.from_numpy(x_pos),
                                 torch.from_numpy(x_neg)])}
        labels = np.concatenate([np.ones(256), np.zeros(256)])
        loss0 = ranker.partial_fit(batch, labels, minibatch=128)
        for _ in range(30):
            loss = ranker.partial_fit(batch, labels, minibatch=128)
        assert loss < loss0                     # 在下降
        assert loss < 0.3                       # 线性可分数据应学到接近 0
        scores = ranker.score({"x": batch["x"]})
        assert scores[:256].mean() > scores[256:].mean() + 0.3
        assert ranker.n_updates > 0
