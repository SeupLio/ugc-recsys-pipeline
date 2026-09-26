#!/usr/bin/env python
"""Step 6：全链路漏斗评测 —— 召回 → 精排 → 重排 → 多样性。

指标口径：
- 漏斗收益：同一个候选集，按不同模型打分重排后的 NDCG@10 / Recall@10 / MAP；
- 生态指标：类目覆盖率、列表内多样性(ILD)、新颖度；MMR 打散后精度的变化；
- 延迟：单用户端到端打分耗时，讨论线上可行性。

用法：
    python scripts/06_eval_funnel.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import (  # noqa: E402
    DIR_CKPT,
    DIR_PROC,
    DIR_RES,
    get_device,
    get_logger,
    save_json,
    seed_everything,
)
from recsys.data.torch_data import build_batch, load_processed  # noqa: E402
from recsys.eval.metrics import (  # noqa: E402
    coverage,
    evaluate_topk,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    recall_at_k,
)
from recsys.models.rank import DeepFMRanker, DINRanker, ESMM, RankConfig  # noqa: E402
from recsys.rerank.mmr import category_cap, cosine_sim_fn, jaccard_sim_fn, mmr_rerank  # noqa: E402

import time  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--cand_topn", type=int, default=80, help="每个用户进入精排的候选数")
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--lambda_div", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_rank_model(kind: str, data, device):
    cfg = RankConfig(
        num_users=data.num_users,
        num_items=data.num_items,
        num_genres=max(data.num_genres, 1),
        num_gender=int(data.users_df["gender_idx"].max()) + 1,
        num_age=int(data.users_df["age_idx"].max()) + 1,
        num_occ=int(data.users_df["occ_idx"].max()) + 1,
    )
    model = {"deepfm": DeepFMRanker, "din": DINRanker, "esmm": ESMM}[kind](cfg).to(device)
    state = torch.load(DIR_CKPT / f"rank_{kind}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def score_candidates(model, data, users, cands, device, kind, batch_size=8192):
    """对每个用户的候选集打分。"""
    all_scores = []
    flat_u, flat_i = [], []
    for u, cs in zip(users, cands):
        flat_u.extend([u] * len(cs))
        flat_i.extend(cs)
    flat_u = np.asarray(flat_u, dtype=np.int64)
    flat_i = np.asarray(flat_i, dtype=np.int64)
    preds = []
    for i in range(0, len(flat_u), batch_size):
        batch = build_batch(data, flat_u[i : i + batch_size], flat_i[i : i + batch_size],
                            device, with_hist=(kind != "deepfm"))
        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]
        preds.append(out.detach().cpu().numpy())
    return flat_u, flat_i, np.concatenate(preds)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    logger = get_logger("funnel")
    device = torch.device(args.device) if args.device != "auto" else get_device()
    proc = Path(args.proc_dir)
    data = load_processed(proc)

    user_emb = np.load(proc / "user_emb.npy")
    item_emb = np.load(proc / "item_emb.npy")
    user_sorted = np.sort(np.load(proc / "fusion_candidates.npy"), axis=0) if False else None

    # ---------------------------------------------------------------- 候选集
    recall_dir = proc
    cand_path = Path(recall_dir) / "fusion_candidates.npy"
    fusion = np.load(cand_path)  # 已按 04 中的用户顺序保存
    test = pd.read_csv(proc / "test_pairs.csv")
    order = sorted(test["user_idx"].unique().tolist())
    gt_map = {int(r.user_idx): int(r.item_idx) for _, r in test.iterrows()}
    idx_of = {u: i for i, u in enumerate(order)}

    users = np.asarray(order, dtype=np.int64)
    cand_list = [fusion[idx_of[u]].tolist()[: args.cand_topn] for u in order]
    # 保证 ground truth 进入候选池：漏召回的样本对精排是「不可学习」的天花板损失
    hit_in_recall = 0
    for i, u in enumerate(order):
        g = gt_map[u]
        if g not in cand_list[i]:
            cand_list[i] = cand_list[i][:-1] + [g]
        else:
            hit_in_recall += 1
    recall_ceiling = hit_in_recall / len(order)
    logger.info("召回层命中率(候选%d)=%.4f —— 这是精排的天然天花板",
                args.cand_topn, recall_ceiling)

    # ------------------------------------------------------------ 精排打分
    rows, latency = [], {}
    preds_by_model = {}
    for kind in ["deepfm", "din", "esmm"]:
        model = load_rank_model(kind, data, device)
        t0 = time.perf_counter()
        flat_u, flat_i, scores = score_candidates(model, data, users, cand_list, device, kind)
        dt = time.perf_counter() - t0
        latency[kind] = float(dt * 1000 / len(order))

        u2score = {}
        start = 0
        for u, cs in zip(users, cand_list):
            u2score[int(u)] = dict(zip(cs, scores[start : start + len(cs)]))
            start += len(cs)
        preds_by_model[kind] = u2score

        u2pred = {int(u): sorted(cand_list[i], key=lambda c: -u2score[int(u)][c])[: args.topn]
                  for i, u in enumerate(order)}
        u2gt = {int(u): [gt_map[int(u)]] for u in order}
        m = evaluate_topk(u2gt, u2pred, ks=(5, 10, 20))
        rows.append({
            "阶段": f"召回+{kind.upper()}",
            "Recall@10": round(m["recall@10"], 4),
            "NDCG@10": round(m["ndcg@10"], 4),
            "MAP@10": round(m["map@10"], 4),
            "单用户打分ms": round(latency[kind], 3),
        })
        logger.info("%s -> %s", kind, {k: round(v, 4) for k, v in m.items()})

    # ------------------------------------------------------------ 重排对比
    genre_map = {int(r.item_idx): json.loads(r.genre_ids) for _, r in data.items_df.iterrows()}
    content_emb = np.load(proc / "content" / "content_emb.npy")
    pop = {int(i): int(c) for i, c in enumerate(data.item_counts)}
    base_u2pred = {int(u): sorted(cand_list[i], key=lambda c: -preds_by_model["din"][int(u)][c])[: args.topn]
                   for i, u in enumerate(order)}
    u2gt = {int(u): [gt_map[int(u)]] for u in order}

    m0 = evaluate_topk(u2gt, base_u2pred, ks=(10,))
    div0 = np.mean([intra_list_diversity(v, genre_map) for v in base_u2pred.values()])
    cov0 = coverage(base_u2pred.values(), data.num_items)
    nov0 = novelty(base_u2pred.values(), pop)
    rows.append({
        "阶段": "DIN重排前Top10",
        "Recall@10": round(m0["recall@10"], 4),
        "NDCG@10": round(m0["ndcg@10"], 4),
        "MAP@10": round(m0["map@10"], 4),
        "ILD(↑越好)": round(float(div0), 4),
        "类目覆盖率(↑)": round(float(cov0), 4),
        "新颖度(↑)": round(nov0, 4),
    })

    sim_fn = cosine_sim_fn(content_emb)
    for lam in [0.9, args.lambda_div, 0.5]:
        u2pred = {}
        for i, u in enumerate(order):
            u = int(u)
            cand_sorted = sorted(cand_list[i], key=lambda c: -preds_by_model["din"][u][c])[: 40]
            u2pred[u] = mmr_rerank(cand_sorted, preds_by_model["din"][u], sim_fn,
                                   topn=args.topn, lambda_div=lam)
        m = evaluate_topk(u2gt, u2pred, ks=(10,))
        div = np.mean([intra_list_diversity(v, genre_map) for v in u2pred.values()])
        rows.append({
            "阶段": f"DIN+MMR(λ={lam})",
            "Recall@10": round(m["recall@10"], 4),
            "NDCG@10": round(m["ndcg@10"], 4),
            "MAP@10": round(m["map@10"], 4),
            "ILD(↑越好)": round(float(div), 4),
            "类目覆盖率(↑)": round(float(coverage(u2pred.values(), data.num_items)), 4),
            "新颖度(↑)": round(novelty(u2pred.values(), pop), 4),
        })

    u2cap = {int(u): category_cap(sorted(cand_list[i],
                                          key=lambda c: -preds_by_model["din"][int(u)][c])[: 40],
                                  genre_map, topn=args.topn, max_per_cat=2)
             for i, u in enumerate(order)}
    m = evaluate_topk(u2gt, u2cap, ks=(10,))
    rows.append({
        "阶段": "DIN+类目打散(≤2/类)",
        "Recall@10": round(m["recall@10"], 4),
        "NDCG@10": round(m["ndcg@10"], 4),
        "MAP@10": round(m["map@10"], 4),
        "ILD(↑越好)": round(float(np.mean([intra_list_diversity(v, genre_map)
                                           for v in u2cap.values()])), 4),
        "类目覆盖率(↑)": round(float(coverage(u2cap.values(), data.num_items)), 4),
        "新颖度(↑)": round(novelty(u2cap.values(), pop), 4),
    })

    out = {
        "recall_ceiling": round(recall_ceiling, 4),
        "cand_topn": args.cand_topn,
        "table": rows,
        "latency_ms_per_user": latency,
    }
    save_json(out, DIR_RES / "funnel_stage.json")
    print("\n=== 全链路漏斗（测试集 %d 用户，候选 %d）===" % (len(order), args.cand_topn))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
