#!/usr/bin/env python
"""Step 4b：ANN 索引的规模效应。

3683 个物品时暴力检索已经够快，ANN 根本没有意义 —— 这个结论本身值得写进报告。
这里把库规模推到 20 万 / 50 万，看看 Flat、IVF、HNSW 的召回保持率与 QPS 怎么变。

用法：
    python scripts/04b_ann_scaling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import DIR_RES, get_logger, save_json  # noqa: E402
from recsys.recall.index import ANNIndex, benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[3_883, 50_000, 200_000])
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--n_query", type=int, default=2000)
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("ann_scaling")
    rng = np.random.default_rng(args.seed)
    rows = []
    for n in args.sizes:
        base = rng.normal(size=(n, args.dim)).astype(np.float32)
        # 人为制造簇结构，让数据更接近真实物品向量分布
        centers = rng.normal(size=(64, args.dim)).astype(np.float32)
        assign = rng.integers(0, 64, size=n)
        vectors = base * 0.35 + centers[assign] * 0.65
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

        queries = vectors[rng.choice(n, size=min(args.n_query, n), replace=False)]
        for name, build in [
            ("Flat", lambda idx: idx.build_flat()),
            ("IVF1024_probe32", lambda idx: idx.build_ivf(nlist=1024, nprobe=32)),
            ("HNSW32_ef64", lambda idx: idx.build_hnsw(m=32, ef_search=64)),
        ]:
            idx = ANNIndex(vectors, metric="ip")
            build(idx)
            b = benchmark(idx, queries, topk=args.topk)
            rows.append({"库规模": n, "索引": name,
                         "召回保持率": round(b["recall_vs_flat"], 4),
                         "QPS": round(b["qps"], 1),
                         "单查询ms": round(b["latency_ms_per_q"], 4)})
            logger.info("%s @ %d -> %s", name, n, b)
    save_json(rows, DIR_RES / "ann_scaling.json")
    print("\n=== ANN 索引规模效应 ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
