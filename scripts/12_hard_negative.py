#!/usr/bin/env python
"""Step 12：难负例挖掘（hard negative mining）及其对精排的影响。

动机：只用 popularity^0.75 采负样本，模型很容易区分「用户看过 vs 完全没听过」，
学不到「用户会看到但不点」的细粒度边界。工业界的常规做法是把**召回靠前但未点击**
的样本作为难负例——它们才是线上真正会曝光的物料。

本脚本实现两阶段挖掘：
1) 用已训练的双塔向量给每个用户的训练正样本做全库检索，取 Top-K；
2) 剔除真实正向交互后，剩下的就是「模型认为很像但用户没点」的难负例；
3) 按难负例比例 ρ 混入原样本重新训练 DIN，与基线（纯流行度负采样）对照。

用法：
    python scripts/12_hard_negative.py --rho 0.5 --epochs 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import (  # noqa: E402
    DIR_PROC,
    DIR_RES,
    get_device,
    get_logger,
    save_json,
    seed_everything,
    timer,
)
from recsys.data.builder import NegativeSampler  # noqa: E402
from recsys.data.torch_data import build_batch, iter_batches, load_processed  # noqa: E402
from recsys.eval.metrics import auc, gauc, log_loss  # noqa: E402
from recsys.models.rank import DINRanker, RankConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--rho", type=float, default=0.5, help="难负例占全部负例的比例")
    p.add_argument("--mine_topk", type=int, default=200, help="挖掘时从召回 Top-K 里取难负例")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_steps", type=int, default=0)
    return p.parse_args()


def mine_hard_negatives(data, user_emb, item_emb, args, logger) -> dict[int, np.ndarray]:
    """为每个用户挖出「双塔认为很像但用户没点」的物品池。

    口径说明：这里只排除**正向交互**（rating>=4）。低分物品（曝光过但用户不喜欢）
    恰恰是最有价值的难负例——它们是带真实曝光信号的负样本；
    而完全未交互的物品属于 unlabeled，作为难负例时按「假定负例」处理（PU learning 视角）。
    """
    positives: dict[int, set[int]] = {}
    pos = data.interactions[data.interactions["rating"] >= 4]
    for uid, grp in pos.groupby("user_idx"):
        positives[int(uid)] = set(grp["item_idx"].tolist())

    users = np.arange(data.num_users)
    hard: dict[int, np.ndarray] = {}
    sims = np.ascontiguousarray(item_emb, dtype=np.float32)
    topk = min(args.mine_topk, data.num_items)
    for u in users:
        s = sims @ np.ascontiguousarray(user_emb[u], dtype=np.float32)
        p = positives.get(int(u), ())
        if p:
            s[list(p)] = -np.inf
        cand = np.argpartition(-s, topk - 1)[:topk]
        hard[int(u)] = cand[np.argsort(-s[cand])]
    logger.info("难负例挖掘完成：%d 个用户，每人 %d 个候选（已排除正向交互）", len(hard), topk)
    return hard


def build_samples(data, hard, args, rng, logger) -> tuple:
    """按 ρ 混合难负例与流行度负例，正样本沿用训练集正向行为。"""
    train_all = data.interactions[data.interactions["split_tag"] == "train"]
    pos = train_all[train_all["rating"] >= 4].sort_values(["user_idx", "ts"]).reset_index(drop=True)
    item_counts = train_all.groupby("item_idx").size().reindex(
        range(data.num_items), fill_value=0).to_numpy()
    sampler = NegativeSampler(data.num_items, item_counts, beta=0.75, seed=args.seed)

    r = 4  # 与主链路一致的 1:4 正负比
    n_hard = int(round(r * args.rho))
    n_soft = r - n_hard

    u_arr = pos["user_idx"].to_numpy()
    i_arr = pos["item_idx"].to_numpy()
    rating = pos["rating"].to_numpy()

    users = np.repeat(u_arr, r + 1)
    items = np.empty(len(users), dtype=np.int64)
    items[:: (r + 1)] = i_arr
    click = np.zeros(len(users), dtype=np.float32)
    click[:: (r + 1)] = 1.0
    convert = np.zeros(len(users), dtype=np.float32)
    convert[:: (r + 1)] = (rating >= 5).astype(np.float32)

    # 难负例：从该用户的 hard 池里按 rank 加权抽样（越靠前越难）
    for j in range(1, n_hard + 1):
        w = np.arange(1, args.mine_topk + 1, dtype=np.float64)
        p = (1.0 / w)
        p /= p.sum()
        pick = np.array([
            rng.choice(hard[int(u)], p=p) for u in u_arr
        ])
        items[j :: (r + 1)] = pick
    # 流行度负例
    for j in range(n_hard + 1, r + 1):
        items[j :: (r + 1)] = sampler.sample_batch(u_arr, {})

    keep = np.repeat(_causal_keep(pos, r + 1), 1)
    logger.info("样本构建: 总 %d，难负例比例 ρ=%.2f（每条正样本 %d 难 / %d 易）",
                len(users), args.rho, n_hard, n_soft)
    return users.astype(np.int64), items.astype(np.int64), click, convert, keep


def _causal_keep(pos: pd.DataFrame, stride: int, max_len: int = 50) -> np.ndarray:
    """与 01_prepare_data 同口径的因果掩码长度。

    序列窗口只保留用户最近 max_len 条正向行为，因此第 j 条样本（用户内 0-based）
    能看到的历史条数 = j - max(0, m - max_len)，其中 m 是该用户的正向行为总数。
    每条正样本及其对应的 stride-1 条负样本必须共享同一个 keep，
    否则模型能从「历史长度」反推标签。
    """
    m = pos.groupby("user_idx")["item_idx"].transform("size").to_numpy()
    j = pos.groupby("user_idx").cumcount().to_numpy()
    start = np.maximum(m - max_len, 0)
    k = np.clip(j - start, 0, np.minimum(m, max_len))
    return np.repeat(k.astype(np.int16), stride)


def make_eval_set(data, seed: int, n_neg: int = 99) -> tuple:
    """易负例评测集：1 正 + 99 流行度负例（与主链路 05 同口径，保证可比）。"""
    valid = data.interactions[data.interactions["split_tag"] == "valid"]
    train_all = data.interactions[data.interactions["split_tag"] == "train"]
    item_counts = train_all.groupby("item_idx").size().reindex(
        range(data.num_items), fill_value=0).to_numpy()
    sampler = NegativeSampler(data.num_items, item_counts, beta=0.75, seed=seed + 1000)
    seen: dict[int, set[int]] = {}
    for uid, grp in train_all.groupby("user_idx"):
        seen[int(uid)] = set(grp["item_idx"].tolist())

    users, items, labels, convs = [], [], [], []
    for _, r in valid.iterrows():
        u, it = int(r["user_idx"]), int(r["item_idx"])
        users.append(u); items.append(it); labels.append(1.0)
        convs.append(1.0 if r["rating"] >= 5 else 0.0)
        negs = sampler.sample(seen.get(u, set()), n_neg)
        users.extend([u] * len(negs)); items.extend(negs)
        labels.extend([0.0] * len(negs)); convs.extend([0.0] * len(negs))
    return (np.asarray(users), np.asarray(items),
            np.asarray(labels, dtype=np.float32), np.asarray(convs, dtype=np.float32))


def make_hard_eval_set(data, hard, seed: int, n_neg: int = 99) -> tuple:
    """难负例评测集：负例取自双塔召回 Top-K 但用户未点。

    这是线上真实曝光分布的近似。只看易负例 AUC 会得出错误结论——
    用难负例训练的模型在易负例口径下分数必然更低（它学会了压低"看着像但不点"的样本），
    但那恰恰是线上需要的能力。两个口径必须一起报。
    """
    valid = data.interactions[data.interactions["split_tag"] == "valid"]
    rng = np.random.default_rng(seed + 2000)
    users, items, labels = [], [], []
    for _, r in valid.iterrows():
        u, it = int(r["user_idx"]), int(r["item_idx"])
        pool = np.array([x for x in hard.get(u, []) if x != it], dtype=np.int64)
        if len(pool) < n_neg:
            continue
        users.append(u); items.append(it); labels.append(1.0)
        negs = rng.choice(pool, size=n_neg, replace=False)
        users.extend([u] * n_neg); items.extend(negs.tolist()); labels.extend([0.0] * n_neg)
    return (np.asarray(users), np.asarray(items), np.asarray(labels, dtype=np.float32))


def train_din(data, samples, args, device, logger, tag: str,
              eval_sets: dict) -> dict:
    users, items, click, convert, keep = samples
    seed_everything(args.seed)
    cfg = RankConfig(
        num_users=data.num_users, num_items=data.num_items,
        num_genres=max(data.num_genres, 1),
        num_gender=int(data.users_df["gender_idx"].max()) + 1,
        num_age=int(data.users_df["age_idx"].max()) + 1,
        num_occ=int(data.users_df["occ_idx"].max()) + 1,
    )
    model = DINRanker(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    bce = nn.BCELoss()

    with timer(f"训练 DIN[{tag}]", logger):
        done = 0
        for ep in range(1, args.epochs + 1):
            model.train()
            rng = np.random.default_rng(args.seed + ep)
            tot, nb = 0.0, 0
            for idx in iter_batches(len(users), args.batch_size, device, rng=rng):
                batch = build_batch(data, users[idx], items[idx], device,
                                    with_hist=True, hist_keep=keep[idx])
                y = torch.as_tensor(click[idx], device=device)
                loss = bce(model(batch), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += float(loss.item()); nb += 1; done += 1
                if args.max_steps and done >= args.max_steps:
                    break
            sched.step()
            logger.info("[%s] epoch %d loss=%.4f", tag, ep, tot / max(nb, 1))
            if args.max_steps and done >= args.max_steps:
                break

    model.eval()
    out = {"tag": tag}
    with torch.no_grad():
        for name, (u, it, y) in eval_sets.items():
            preds = []
            for i in range(0, len(u), 16384):
                b = build_batch(data, u[i:i + 16384], it[i:i + 16384], device, with_hist=True)
                preds.append(model(b).detach().cpu().numpy())
            P = np.concatenate(preds)
            out[f"auc_{name}"] = auc(y, P)
            out[f"gauc_{name}"] = gauc(y, P, u)
            out[f"logloss_{name}"] = log_loss(y, P)
            logger.info("[%s] %-10s AUC=%.4f GAUC=%.4f", tag, name,
                        out[f"auc_{name}"], out[f"gauc_{name}"])
    return out


def main() -> None:
    args = parse_args()
    logger = get_logger("hard_neg")
    device = torch.device(args.device) if args.device != "auto" else get_device()
    proc = Path(args.proc_dir)
    data = load_processed(proc)

    user_emb = np.load(proc / "user_emb.npy")
    item_emb = np.load(proc / "item_emb.npy")

    hard = mine_hard_negatives(data, user_emb, item_emb, args, logger)

    easy_u, easy_i, easy_y, _ = make_eval_set(data, args.seed)
    hard_u, hard_i, hard_y = make_hard_eval_set(data, hard, args.seed)
    eval_sets = {
        "easy": (easy_u, easy_i, easy_y),
        "hard": (hard_u, hard_i, hard_y),
    }
    logger.info("评测集：易负例 %d 条 / 难负例 %d 条", len(easy_u), len(hard_u))

    rng = np.random.default_rng(args.seed)
    results = []

    # 基线：ρ=0（纯流行度负例），与主链路口径一致
    base_args = argparse.Namespace(**{**vars(args), "rho": 0.0})
    base = build_samples(data, hard, base_args, rng, logger)
    results.append(train_din(data, base, args, device, logger, "baseline ρ=0", eval_sets))

    # 实验组：混合难负例
    mix = build_samples(data, hard, args, rng, logger)
    results.append(train_din(data, mix, args, device, logger, f"hard-neg ρ={args.rho}", eval_sets))

    rows = [{
        "负采样策略": r["tag"],
        "易负例AUC": round(r["auc_easy"], 4),
        "难负例AUC": round(r["auc_hard"], 4),
        "难负例GAUC": round(r["gauc_hard"], 4),
        "难负例LogLoss": round(r["logloss_hard"], 4),
    } for r in results]
    save_json({"rho": args.rho, "mine_topk": args.mine_topk, "epochs": args.epochs,
               "results": results}, DIR_RES / "hard_negative.json")
    print("\n=== 难负例挖掘对精排的影响 ===")
    print("说明：两个口径必须一起看。用难负例训练会压低易负例口径的分数，")
    print("      但难负例口径更贴近线上真实曝光分布。")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
