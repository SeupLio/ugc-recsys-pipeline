#!/usr/bin/env python
"""Step 13：意图标签召回 + 质量感知重排（内容理解如何真正接入链路）。

Step 11 产出了两个内容理解资产：受控标签的用户意图分布、物品的贝叶斯平滑质量分。
本脚本验证它们**能不能真的换到链路收益**，而不是停留在"看起来有用"：

1) **意图标签召回**：用户意图分布 × 物品标签 → 打分召回。它不依赖行为共现，
   因此对行为稀疏用户和新内容都有效，是 ItemCF/双塔之外的第四路。
2) **质量感知重排**：把质量分作为重排的先验项（低质内容降权）。
   社区场景里这是生态治理的常规手段：不是不让低质内容曝光，而是不让它占据头部坑位。

用法：
    python scripts/13_intent_rerank.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import DIR_PROC, DIR_RES, get_logger, save_json  # noqa: E402
from recsys.data.torch_data import load_processed  # noqa: E402
from recsys.eval.metrics import (  # noqa: E402
    coverage,
    evaluate_topk,
    intra_list_diversity,
    novelty,
)
from recsys.recall.collaborative import merge_multi_way  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--quality_alpha", type=float, default=0.3,
                   help="重排时质量分的权重：score = pCTR - α·penalty(low quality)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("intent_rerank")
    proc = Path(args.proc_dir)
    data = load_processed(proc)

    tags = json.loads((proc / "content" / "content_tags.json").read_text(encoding="utf-8"))
    tag_names = json.loads((proc / "intent_tags.json").read_text(encoding="utf-8"))["tag_names"]
    intent = np.load(proc / "intent_matrix.npy")
    quality = np.load(proc / "quality_score.npy")

    tag_idx = {t: i for i, t in enumerate(tag_names)}
    # 物品 → 标签 one-hot 矩阵
    item_tag = np.zeros((data.num_items, len(tag_names)), dtype=np.float32)
    for it_str, ts in tags.items():
        it = int(it_str)
        for t in ts:
            if t in tag_idx:
                item_tag[it, tag_idx[t]] = 1.0

    # ------------------------------------------------------------ 评测集
    test = pd.read_csv(proc / "test_pairs.csv")
    gt = {int(r.user_idx): [int(r.item_idx)] for _, r in test.iterrows()}
    users = sorted(gt)
    train_all = data.interactions[data.interactions["split_tag"] == "train"]
    seen: dict[int, set[int]] = {}
    for uid, grp in train_all.groupby("user_idx"):
        seen[int(uid)] = set(grp["item_idx"].tolist())

    cold_items = set(np.load(proc / "cold_items.npy").tolist())
    warm_pool = np.array([i for i in range(data.num_items) if i not in cold_items])
    warm_pos = {int(it): p for p, it in enumerate(warm_pool)}  # item_idx -> warm_pool 位置

    # ------------------------------------------------- 1) 意图标签召回（第四路）
    logger.info("意图标签召回：用户意图向量 × 物品标签矩阵")
    scores_all = intent @ item_tag[warm_pool].T  # (U, warm)
    intent_pred = {}
    for u in users:
        s = scores_all[u].copy()
        # seen 里是 item_idx，要先映射到 warm_pool 的位置再屏蔽
        seen_pos = [warm_pos[i] for i in seen.get(u, ()) if i in warm_pos]
        if seen_pos:
            s[seen_pos] = -np.inf
        k = min(args.topk, len(warm_pool))
        cand = np.argpartition(-s, k - 1)[:k]
        intent_pred[u] = warm_pool[cand[np.argsort(-s[cand])]].tolist()
    m_intent = evaluate_topk(gt, intent_pred, ks=(10, 20, 50))
    logger.info("意图标签召回(单独): %s", {k: round(v, 4) for k, v in m_intent.items()})

    # 与已有三路融合，看第四路是否带来增量
    fusion = np.load(proc / "fusion_candidates.npy")
    test_users_sorted = sorted(test["user_idx"].unique().tolist())
    idx_of = {u: i for i, u in enumerate(test_users_sorted)}
    base_pred = {u: fusion[idx_of[u]].tolist()[: args.topk] for u in users}
    m_base = evaluate_topk(gt, base_pred, ks=(10, 20, 50))

    fused_pred = {
        u: merge_multi_way(
            {"base": base_pred[u], "intent": intent_pred[u]},
            {"base": 1.0, "intent": 0.5},
            topn=args.topk,
        )
        for u in users
    }
    m_fused = evaluate_topk(gt, fused_pred, ks=(10, 20, 50))
    logger.info("三路融合(基线): Recall@50=%.4f | +意图第四路: Recall@50=%.4f",
                m_base["recall@50"], m_fused["recall@50"])

    # 意图召回对新内容的能力（它不依赖行为，理论上对冷启动物品也有效）
    cold_pool = np.array(sorted(cold_items))
    cold_scores = intent @ item_tag[cold_pool].T
    cold_pred = {}
    cold_gt = {}
    cold_pairs = pd.read_csv(proc / "cold_pairs.csv")
    for _, r in cold_pairs.iterrows():
        cold_gt.setdefault(int(r["user_idx"]), []).append(int(r["item_idx"]))
    for u in cold_gt:
        if u >= len(cold_scores):
            continue
        s = cold_scores[u]
        k = min(args.topk, len(cold_pool))
        cand = np.argpartition(-s, k - 1)[:k]
        cold_pred[u] = cold_pool[cand[np.argsort(-s[cand])]].tolist()
    m_cold = evaluate_topk(cold_gt, cold_pred, ks=(20, 50)) if cold_pred else {}
    logger.info("意图召回在冷启动物品上: %s", {k: round(v, 4) for k, v in m_cold.items()})

    # ------------------------------------------------- 2) 质量感知重排
    genre_map = {int(r.item_idx): json.loads(r.genre_ids) for _, r in data.items_df.iterrows()}
    pop = {int(i): int(c) for i, c in enumerate(data.item_counts)}

    # 用双塔分数模拟精排打分（避免重复加载 DIN；重点是质量项的边际作用）
    user_emb = np.load(proc / "user_emb.npy")
    item_emb = np.load(proc / "item_emb.npy")

    q_z = (quality - quality.mean()) / (quality.std() + 1e-8)
    rows = []
    TOPN = 10  # 列表级指标必须在最终展示的 Top10 上算，50 条候选集与 α 无关
    for alpha in [0.0, args.quality_alpha, 0.6]:
        u2pred = {}
        for u in users:
            cand = np.asarray(base_pred[u], dtype=np.int64)
            rel = item_emb[cand] @ user_emb[u]
            # 质量分只惩罚低质，不奖励高质（避免把榜单变成"经典电影榜"）
            penalty = np.clip(-q_z[cand], 0, None)
            score = rel - alpha * penalty
            order = cand[np.argsort(-score)][:TOPN]
            u2pred[u] = order.tolist()
        m = evaluate_topk(gt, u2pred, ks=(10,))
        div = float(np.mean([intra_list_diversity(v, genre_map) for v in u2pred.values()]))
        avg_q = float(np.mean([quality[i] for v in u2pred.values() for i in v]))
        rows.append({
            "质量权重α": alpha,
            "Recall@10": round(m["recall@10"], 4),
            "NDCG@10": round(m["ndcg@10"], 4),
            "ILD@10↑": round(div, 4),
            "覆盖率↑": round(float(coverage(u2pred.values(), data.num_items)), 4),
            "新颖度↑": round(novelty(u2pred.values(), pop), 4),
            "Top10平均质量分": round(avg_q, 4),
        })
        logger.info("α=%.2f -> %s", alpha, rows[-1])

    out = {
        "intent_channel_alone": {k: round(v, 4) for k, v in m_intent.items()},
        "baseline_three_way": {k: round(v, 4) for k, v in m_base.items()},
        "four_way_fusion": {k: round(v, 4) for k, v in m_fused.items()},
        "intent_on_cold_items": {k: round(v, 4) for k, v in m_cold.items()},
        "quality_rerank": rows,
    }
    save_json(out, DIR_RES / "intent_quality_stage.json")

    print("\n=== 意图标签召回（第四路）===")
    print(pd.DataFrame([
        {"通道": "三路融合(基线)", **{k: round(v, 4) for k, v in m_base.items() if k.startswith(("recall", "ndcg"))}},
        {"通道": "意图标签(单独)", **{k: round(v, 4) for k, v in m_intent.items() if k.startswith(("recall", "ndcg"))}},
        {"通道": "四路融合", **{k: round(v, 4) for k, v in m_fused.items() if k.startswith(("recall", "ndcg"))}},
    ]).to_string(index=False))
    print("\n=== 质量感知重排 ===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
