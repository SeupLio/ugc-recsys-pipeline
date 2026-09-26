"""召回 / 重排 / 索引 的行为测试。"""

from __future__ import annotations

import numpy as np
import pytest

from recsys.rerank.mmr import category_cap, cosine_sim_fn, jaccard_sim_fn, mmr_rerank
from recsys.recall.collaborative import ItemCF, PopularityRecall, merge_multi_way


def _interactions():
    # 用户 0、1 共同喜欢 0/1；用户 2 喜欢 2
    return type("D", (), {})(), {
        "user_idx": [0, 0, 1, 1, 2, 2],
        "item_idx": [0, 1, 0, 1, 2, 1],
        "ts": [10, 11, 12, 13, 14, 15],
    }


def test_itemcf_recalls_cooccurrence_item():
    import pandas as pd

    df = pd.DataFrame({"user_idx": [0, 0, 1, 1, 2], "item_idx": [0, 1, 0, 1, 2],
                       "ts": [1, 2, 3, 4, 5]})
    icf = ItemCF(num_items=4, top_k_sim=3).fit(df)
    rec = icf.recall([0], topn=3)
    assert 0 not in rec, "已看过的物品不应再被推荐"
    assert 1 in rec, "共现最强的物品应当被召回"


def test_itemcf_empty_history_returns_empty():
    import pandas as pd

    df = pd.DataFrame({"user_idx": [0], "item_idx": [0], "ts": [1]})
    icf = ItemCF(num_items=3, top_k_sim=2).fit(df)
    assert icf.recall([], topn=2) == []


def test_popularity_recall_prefers_recent():
    import pandas as pd

    df = pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [0, 1, 0], "ts": [1, 1000, 1000]})
    hot = PopularityRecall.from_interactions(df, num_items=3, half_life_days=0.001)
    rec = hot.recall(topn=2, exclude=[])
    assert rec[0] == 0, "时间衰减后更近的物品应排在前面"


def test_popularity_recall_respects_exclude():
    import pandas as pd

    df = pd.DataFrame({"user_idx": [0], "item_idx": [0], "ts": [1]})
    hot = PopularityRecall.from_interactions(df, num_items=3)
    assert 0 not in hot.recall(topn=3, exclude=[0])


def test_merge_multi_way_rrf_prefers_high_rank_items():
    a = [10, 11, 12]
    b = [11, 13, 14]
    merged = merge_multi_way({"a": a, "b": b}, {"a": 1.0, "b": 1.0}, topn=3)
    assert merged[0] == 11, "两路都排在前面的物品应获得最高融合分"


def test_merge_multi_way_dedups():
    merged = merge_multi_way({"a": [1, 2], "b": [1, 2, 3]}, topn=5)
    assert len(merged) == len(set(merged))


def test_mmr_increases_diversity():
    sim_fn = jaccard_sim_fn({0: [1, 2], 1: [1, 2], 2: [3], 3: [4]})
    scores = {0: 0.9, 1: 0.89, 2: 0.5, 3: 0.4}
    greedy = mmr_rerank([0, 1, 2, 3], scores, sim_fn, topn=3, lambda_div=1.0)  # 纯相关性
    diverse = mmr_rerank([0, 1, 2, 3], scores, sim_fn, topn=3, lambda_div=0.2)  # 强多样性
    assert diverse != greedy
    assert 2 in diverse, "打散后应当让不同类目的内容进入列表"


def test_mmr_returns_unique_items():
    sim_fn = jaccard_sim_fn({i: [i % 3] for i in range(10)})
    out = mmr_rerank(list(range(10)), {i: float(i) for i in range(10)}, sim_fn, topn=5)
    assert len(out) == 5 == len(set(out))


def test_category_cap_limits_per_category():
    feat = {0: [1], 1: [1], 2: [1], 3: [2], 4: [2], 5: [3]}
    out = category_cap([0, 1, 2, 3, 4, 5], feat, topn=5, max_per_cat=2)
    counts = {}
    for it in out:
        for c in feat[it]:
            counts[c] = counts.get(c, 0) + 1
    assert max(counts.values()) <= 2


def test_cosine_sim_fn_normalized_input():
    v = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    fn = cosine_sim_fn(v)
    assert abs(fn(0, 0) - 1.0) < 1e-6
    assert abs(fn(0, 1)) < 1e-6
    assert abs(fn(0, 2) - 1.0) < 1e-6


def test_ann_flat_self_recall():
    pytest.importorskip("faiss")
    from recsys.recall.index import ANNIndex, benchmark

    rng = np.random.default_rng(0)
    vec = rng.normal(size=(200, 16)).astype(np.float32)
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
    idx = ANNIndex(vec, metric="ip").build_flat()
    b = benchmark(idx, vec[:50], topk=10)
    assert b["recall_vs_flat"] > 0.999
    assert b["qps"] > 0


def test_ann_ivf_not_worse_than_random():
    pytest.importorskip("faiss")
    from recsys.recall.index import ANNIndex, recall_overlap

    rng = np.random.default_rng(1)
    vec = rng.normal(size=(2000, 32)).astype(np.float32)
    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
    idx = ANNIndex(vec, metric="ip").build_ivf(nlist=20, nprobe=4)
    _, approx = idx.search(vec[:100], 20)
    _, gt = idx.search_flat(vec[:100], 20)
    assert recall_overlap(approx, gt) > 0.5, "IVF 至少应保留一半以上的暴力召回结果"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
