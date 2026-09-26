#!/usr/bin/env python
"""Step 9：单次请求端到端 Demo（召回 → 精排 → 重排）。

把「一次推荐请求在系统内部发生了什么」完整地打出来：
每路召回的候选数、融合后的候选数、精排打分延迟、重排前后的类目分布变化。

用法：
    python scripts/09_demo_serve.py --user 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import DIR_CKPT, DIR_PROC, get_device, get_logger  # noqa: E402
from recsys.data.torch_data import build_batch, load_processed  # noqa: E402
from recsys.models.rank import DINRanker, RankConfig  # noqa: E402
from recsys.recall.collaborative import ItemCF, PopularityRecall  # noqa: E402
from recsys.rerank.mmr import mmr_rerank  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--user", type=int, default=7)
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--candidates", type=int, default=50)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("serve")
    proc = Path(args.proc_dir)
    data = load_processed(proc)
    device = torch.device(args.device) if args.device != "auto" else get_device()

    user_emb = np.load(proc / "user_emb.npy")
    item_emb = np.load(proc / "item_emb.npy")
    content = np.load(proc / "content" / "content_emb.npy")
    titles = dict(zip(data.items_df["item_idx"], data.items_df["clean_title"]))
    genres = {int(r.item_idx): json.loads(r.genre_ids) for _, r in data.items_df.iterrows()}
    gname = {v: k for k, v in data.vocab["genre2idx"].items()}

    u = int(args.user)
    hist = [int(i) for i in data.seq_mat[u][data.seq_mat[u] >= 0].numpy()]
    seen = set(hist)
    logger.info("用户 #%d，历史上看过 %d 条内容", u, len(hist))

    train_all = data.interactions[data.interactions["split_tag"] == "train"]
    train_pos = train_all[train_all["rating"] >= 4]

    # ------------------------------------------------------------- 召回
    t0 = time.perf_counter()
    hot = PopularityRecall.from_interactions(train_pos, data.num_items, half_life_days=30.0)
    hot_list = hot.recall(topn=args.candidates, exclude=seen)
    t_hot = time.perf_counter() - t0

    t0 = time.perf_counter()
    icf = ItemCF(data.num_items, top_k_sim=100).fit(train_pos)
    icf_list = [i for i in icf.recall(hist, topn=args.candidates) if i not in seen]
    t_icf = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = item_emb @ user_emb[u]
    scores[list(seen)] = -np.inf
    cand = np.argpartition(-scores, args.candidates)[: args.candidates]
    tt_list = cand[np.argsort(-scores[cand])].tolist()
    t_tt = time.perf_counter() - t0

    fused = list(dict.fromkeys(hot_list[:20] + icf_list[:30] + tt_list[:40]))[: args.candidates]
    logger.info("召回: 热度=%d ItemCF=%d 双塔=%d → 融合候选=%d | 耗时 %.1fms/%.1fms/%.1fms",
                len(hot_list), len(icf_list), len(tt_list), len(fused),
                t_hot * 1000, t_icf * 1000, t_tt * 1000)

    # ------------------------------------------------------------- 精排
    cfg = RankConfig(
        num_users=data.num_users, num_items=data.num_items,
        num_genres=max(data.num_genres, 1),
        num_gender=int(data.users_df["gender_idx"].max()) + 1,
        num_age=int(data.users_df["age_idx"].max()) + 1,
        num_occ=int(data.users_df["occ_idx"].max()) + 1,
    )
    model = DINRanker(cfg).to(device)
    state = torch.load(DIR_CKPT / "rank_din.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    candidates = np.asarray(fused, dtype=np.int64)
    users = np.full(len(candidates), u, dtype=np.int64)
    t0 = time.perf_counter()
    with torch.no_grad():
        batch = build_batch(data, users, candidates, device, with_hist=True)
        preds = model(batch).detach().cpu().numpy()
    t_rank = time.perf_counter() - t0
    score_map = {int(c): float(p) for c, p in zip(candidates, preds)}
    ranked = sorted(fused, key=lambda c: -score_map[c])
    logger.info("精排: %d 条候选，单请求 %.2fms", len(fused), t_rank * 1000)

    # ------------------------------------------------------------- 重排
    def cos_sim(a: int, b: int) -> float:
        return float(np.dot(content[a], content[b]))

    t0 = time.perf_counter()
    final = mmr_rerank(ranked[:40], score_map, cos_sim, topn=args.topn, lambda_div=0.7)
    t_rr = time.perf_counter() - t0

    def describe(items):
        rows = []
        for rank, it in enumerate(items, 1):
            gs = [gname.get(g, str(g)) for g in genres.get(int(it), [])]
            rows.append({
                "rank": rank, "item": int(it), "title": str(titles.get(int(it), "?")),
                "genres": "/".join(gs),
                "score": round(score_map.get(int(it), float("nan")), 4),
            })
        return rows

    before, after = describe(ranked[: args.topn]), describe(final)
    print("\n=== 重排前（纯精排 Top%d）===" % args.topn)
    print(pd.DataFrame(before).to_string(index=False))
    print("\n=== 重排后（MMR λ=0.7 Top%d，耗时 %.2fms）===" % (args.topn, t_rr * 1000))
    print(pd.DataFrame(after).to_string(index=False))
    print("\n总耗时: 召回 %.1fms + 精排 %.1fms + 重排 %.1fms = %.1fms" %
          (t_hot * 1000 + t_icf * 1000 + t_tt * 1000, t_rank * 1000, t_rr * 1000,
           t_hot * 1000 + t_icf * 1000 + t_tt * 1000 + t_rank * 1000 + t_rr * 1000))


if __name__ == "__main__":
    main()
