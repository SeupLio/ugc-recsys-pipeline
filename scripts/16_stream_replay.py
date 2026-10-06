#!/usr/bin/env python
"""Step 16：真实事件流回放 + 在线增量学习 + 先序评测（无泄漏）。

与离线评测的根本差异：这里的一切都发生在真实时间轴上。

1. 真实流量源：ML-1M 的 98.8 万条评分带真实时间戳（跨 1039 天），
   按时间排序成事件流——流行度漂移、活跃用户突发、长尾稀疏全是
   数据本身的特性，不是模拟器生成的；
2. 时间切分：前 80%（按时间）热身训练，后 20% 作为流；
3. 先序评测（test-then-train）：每个窗口先用「未见过该窗口」的当前
   模型打分（AUC 无泄漏），再 partial_fit 增量更新——量化
   「离线 AUC」与「时间轴上的真实表现」之间的差距；
4. 在线组件同步更新：时间衰减热度榜（含 top-100 换手率）、
   分数分布 PSI 漂移；
5. 产出 results/stream_replay.json（逐窗 AUC 轨迹 / 漂移 / 热度换手）。

用法：
    python scripts/16_stream_replay.py [--window 20000] [--model deepfm]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from recsys.common import DIR_PROC, ROOT, get_logger, save_json  # noqa: E402
from recsys.data.torch_data import build_batch, load_processed  # noqa: E402
from recsys.models.rank import DeepFMRanker, RankConfig  # noqa: E402
from recsys.serving.stream import (  # noqa: E402
    DecayedPopularity, OnlineRanker, PrequentialReplayer, load_event_stream)

log = get_logger("stream_replay")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=3, help="热身训练轮数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t0 = time.perf_counter()
    data = load_processed(DIR_PROC)
    users, items, ts = load_event_stream(DIR_PROC / "interactions.csv")
    log.info("加载 %d 条真实事件（%d 天跨度），窗口 %d",
             len(users), (ts[-1] - ts[0]) / 86400, args.window)

    # —— 模型：DeepFM（与主链路同结构、同特征），从零训练（不用旧
    #    checkpoint——它是在含未来数据的随机切分上训的，会污染时间轴）
    cfg = RankConfig(
        num_users=data.num_users, num_items=data.num_items,
        num_genres=max(data.num_genres, 1),
        num_gender=int(data.users_df["gender_idx"].max()) + 1,
        num_age=int(data.users_df["age_idx"].max()) + 1,
        num_occ=int(data.users_df["occ_idx"].max()) + 1)
    model = DeepFMRanker(cfg)
    ranker = OnlineRanker(model, lr=5e-4)

    # —— 热身：前 80%（按时间）负采样训练 ——
    cut = int(len(users) * 0.8)
    rng = np.random.default_rng(args.seed)
    pop_all = data.item_counts.astype(np.float64) ** 0.75
    pop_all /= pop_all.sum()
    from recsys.serving.stream import UserHistory
    hist_w = UserHistory(data.num_users)
    u_w, i_w = users[:cut], items[:cut]
    hist_w.add(u_w, i_w)
    neg_w = hist_w.sample_negatives(u_w, rng, data.num_items, pop_p=pop_all)
    labels = np.concatenate([np.ones(len(u_w)), np.zeros(len(u_w))])
    batch_w = build_batch(data, np.concatenate([u_w, u_w]),
                          np.concatenate([i_w, neg_w]))
    for ep in range(args.epochs):
        loss = ranker.partial_fit(batch_w, labels, n_steps=1)
        log.info("热身 epoch %d/%d loss=%.4f", ep + 1, args.epochs, loss)

    # —— 先序回放：后 20% 流（replayer 自建热度/历史/优化器，权重沿用热身模型）——
    replayer = PrequentialReplayer(
        model=model, num_items=data.num_items, num_users=data.num_users,
        window=args.window, lr=5e-4, half_life_s=180 * 86400.0, seed=args.seed)

    def _build(users_, items_):
        return build_batch(data, users_, items_)

    reports = replayer.run(
        users[cut:], items[cut:], ts[cut:], build_batch=_build, warmup_frac=0.0,
        on_window=lambda r: log.info(
            "窗口 %d: AUC=%.4f（热度基线 %.3f）PSI=%.3f 换手率=%.3f loss=%.3f",
            r.idx, r.auc, r.auc_pop, r.psi, r.top_churn, r.loss))

    aucs = [r.auc for r in reports]
    out = {
        "n_events": int(len(users)), "n_stream": int(len(users) - cut),
        "window": args.window, "warmup_events": int(cut),
        "epochs": args.epochs,
        "mean_auc": round(float(np.mean(aucs)), 4),
        "first_auc": round(aucs[0], 4), "last_auc": round(aucs[-1], 4),
        "auc_delta": round(aucs[-1] - aucs[0], 4),
        "mean_pop_auc": round(float(np.mean([r.auc_pop for r in reports])), 4),
        "n_windows": len(reports),
        "drift_windows": [r.idx for r in reports if r.drift],
        "mean_churn": round(float(np.mean([r.top_churn for r in reports])), 3),
        "windows": [asdict(r) for r in reports],
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    save_json(out, ROOT / "results" / "stream_replay.json")
    log.info("完成：均值 AUC %.4f（首窗 %.4f → 末窗 %.4f，增量 Δ%.4f）"
             "热度基线 %.4f，均值换手率 %.3f",
             out["mean_auc"], out["first_auc"], out["last_auc"],
             out["auc_delta"], out["mean_pop_auc"], out["mean_churn"])
    print(json.dumps({k: v for k, v in out.items() if k != "windows"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
