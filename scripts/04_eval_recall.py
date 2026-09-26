#!/usr/bin/env python
"""Step 4：召回层评测 —— 多路召回对比 + ANN 索引权衡。

对比通道：热度榜 / ItemCF / 双塔向量 / 双塔 + ItemCF 融合（RRF）。
另外给出 Flat vs IVF vs HNSW 的召回保持率与 QPS，回答「近似检索到底亏了多少」。

用法：
    python scripts/04_eval_recall.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import DIR_PROC, DIR_RES, get_logger, save_json, timer  # noqa: E402
from recsys.data.torch_data import load_processed  # noqa: E402
from recsys.eval.metrics import evaluate_topk  # noqa: E402
from recsys.recall.collaborative import ItemCF, PopularityRecall, merge_multi_way  # noqa: E402
from recsys.recall.index import ANNIndex, benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--fusion_topn", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("recall_eval")
    proc = Path(args.proc_dir)
    data = load_processed(proc)

    user_emb = np.load(proc / "user_emb.npy")
    item_emb = np.load(proc / "item_emb.npy")

    test = pd.read_csv(proc / "test_pairs.csv")
    train_all = data.interactions[data.interactions["split_tag"] == "train"]
    train_pos = train_all[train_all["rating"] >= 4]

    gt = {int(r.user_idx): [int(r.item_idx)] for _, r in test.iterrows()}
    seen: dict[int, set[int]] = {}
    for uid, grp in train_all.groupby("user_idx"):
        seen[int(uid)] = set(grp["item_idx"].tolist())

    results: dict[str, dict] = {}

    # ---------------------------------------------------------- 1. 热度榜
    with timer("热度榜召回", logger):
        hot = PopularityRecall.from_interactions(train_pos, data.num_items, half_life_days=30.0)
        hot_pred = {u: hot.recall(topn=args.topk, exclude=seen.get(u, [])) for u in gt}
    results["热度榜(30天半衰期)"] = evaluate_topk(gt, hot_pred, ks=(10, 20, 50))

    # ---------------------------------------------------------- 2. ItemCF
    with timer("ItemCF 召回", logger):
        icf = ItemCF(data.num_items, top_k_sim=100, half_life_days=180.0).fit(train_pos)
        hist_map = {int(u): list(data.seq_mat[u][data.seq_mat[u] >= 0].numpy()) for u in gt}
        icf_pred = {
            u: [i for i in icf.recall(hist_map.get(u, []), topn=args.topk)
                if i not in seen.get(u, set())][: args.topk]
            for u in gt
        }
    results["ItemCF(IUF+Top100)"] = evaluate_topk(gt, icf_pred, ks=(10, 20, 50))

    # ---------------------------------------------------- 3. 双塔暴力检索
    with timer("双塔向量召回(Flat)", logger):
        index = ANNIndex(item_emb, metric="ip").build_flat()
        q = np.ascontiguousarray(user_emb, dtype=np.float32)
        _, flat_ids = index.search_flat(q, args.topk * 3)
        tt_pred = {}
        for pos, u in enumerate(range(data.num_users)):
            if u not in gt:
                continue
            cand = [int(i) for i in flat_ids[pos] if i not in seen.get(u, set())]
            tt_pred[u] = cand[: args.topk]
    results["双塔(暴力检索)"] = evaluate_topk(gt, tt_pred, ks=(10, 20, 50))

    # ---------------------------------------------------------- 4. 融合 RRF
    fusion_pred = {}
    for u in gt:
        ways = {
            "hot": hot.recall(topn=args.fusion_topn, exclude=seen.get(u, [])),
            "itemcf": [i for i in icf.recall(hist_map.get(u, []), topn=args.fusion_topn)
                       if i not in seen.get(u, set())],
            "two_tower": tt_pred.get(u, []),
        }
        fusion_pred[u] = merge_multi_way(ways, {"hot": 0.6, "itemcf": 1.0, "two_tower": 1.0},
                                         topn=args.topk)
    results["三路融合(RRF)"] = evaluate_topk(gt, fusion_pred, ks=(10, 20, 50))

    # ------------------------------------------------------------ 5. ANN 权衡
    ann_bench = {}
    for name, builder in [
        ("Flat", lambda idx: idx.build_flat()),
        ("IVF100_probe10", lambda idx: idx.build_ivf(nlist=100, nprobe=10)),
        ("IVF256_probe32", lambda idx: idx.build_ivf(nlist=256, nprobe=32)),
        ("HNSW32_ef64", lambda idx: idx.build_hnsw(m=32, ef_search=64)),
    ]:
        idx = ANNIndex(item_emb, metric="ip")
        builder(idx)
        q_users = np.ascontiguousarray(user_emb, dtype=np.float32)
        with timer(f"{name} benchmark", None):
            bench = benchmark(idx, q_users, topk=args.topk)
        ann_bench[name] = bench
        logger.info("%s -> %s", name, bench)

    # IVF 真实召回效果（带 seen 过滤）
    idx_ivf = ANNIndex(item_emb, metric="ip").build_ivf(nlist=100, nprobe=20)
    _, ivf_ids = idx_ivf.search(np.ascontiguousarray(user_emb, dtype=np.float32), args.topk * 3)
    ivf_pred = {
        u: [int(i) for i in ivf_ids[pos] if i not in seen.get(u, set())][: args.topk]
        for pos, u in enumerate(range(data.num_users))
        if u in gt
    }
    results["双塔(IVF100_probe20)"] = evaluate_topk(gt, ivf_pred, ks=(10, 20, 50))

    # ---------------------------------------------------------------- 落盘
    table = []
    for name, m in results.items():
        table.append(
            {
                "召回通道": name,
                "Recall@10": round(m["recall@10"], 4),
                "Recall@20": round(m["recall@20"], 4),
                "Recall@50": round(m["recall@50"], 4),
                "NDCG@20": round(m["ndcg@20"], 4),
                "HitRate@50": round(m["hit@50"], 4),
            }
        )
    out = {
        "table": table,
        "ann_benchmark": ann_bench,
        "num_test_users": len(gt),
        "topk": args.topk,
    }
    save_json(out, DIR_RES / "recall_stage.json")
    print("\n=== 召回层评测（测试集 %d 用户）===" % len(gt))
    print(pd.DataFrame(table).to_string(index=False))
    print("\n=== ANN 索引权衡 ===")
    print(pd.DataFrame(ann_bench).T.to_string())

    # 保存融合结果，供排序层使用
    np.save(proc / "fusion_candidates.npy", np.array(
        [fusion_pred[u] for u in sorted(fusion_pred)], dtype=np.int64))


if __name__ == "__main__":
    main()
