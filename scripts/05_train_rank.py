#!/usr/bin/env python
"""Step 5：精排模型训练与对比（DeepFM / DIN / ESMM 多目标）。

评估口径统一：每个 valid 用户取「留一法正向物品 + 99 条流行度负样本」，
指标为 AUC / GAUC / LogLoss；ESMM 额外报告 CTCVR 的 AUC。

用法：
    python scripts/05_train_rank.py --model all --epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
    timer,
)
from recsys.data.builder import NegativeSampler  # noqa: E402
from recsys.data.torch_data import build_batch, iter_batches, load_processed  # noqa: E402
from recsys.eval.metrics import auc, gauc, log_loss  # noqa: E402
from recsys.models.rank import DeepFMRanker, DINRanker, ESMM, RankConfig, esmm_loss  # noqa: E402

MODEL_CLS = {"deepfm": DeepFMRanker, "din": DINRanker, "esmm": ESMM}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--model", type=str, default="all", choices=["all", "deepfm", "din", "esmm"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=0, help=">0 时只跑指定步数，用于冒烟测试")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def build_rank_config(data) -> RankConfig:
    return RankConfig(
        num_users=data.num_users,
        num_items=data.num_items,
        num_genres=max(data.num_genres, 1),
        num_gender=int(data.users_df["gender_idx"].max()) + 1,
        num_age=int(data.users_df["age_idx"].max()) + 1,
        num_occ=int(data.users_df["occ_idx"].max()) + 1,
    )


def make_eval_set(data, seed: int, n_neg: int = 99) -> tuple:
    """构造「1 正 + 99 负」的排序评测集，负样本按流行度抽取并排除已看物品。"""
    valid = data.interactions[data.interactions["split_tag"] == "valid"]
    train_all = data.interactions[data.interactions["split_tag"] == "train"]
    item_counts = train_all.groupby("item_idx").size().reindex(
        range(data.num_items), fill_value=0).to_numpy()
    sampler = NegativeSampler(data.num_items, item_counts, beta=0.75, seed=seed + 1000)
    seen: dict[int, set[int]] = {}
    for uid, grp in data.interactions[data.interactions["split_tag"] == "train"].groupby("user_idx"):
        seen[int(uid)] = set(grp["item_idx"].tolist())

    users, items, labels, converts = [], [], [], []
    for _, r in valid.iterrows():
        u, it = int(r["user_idx"]), int(r["item_idx"])
        users.append(u)
        items.append(it)
        labels.append(1.0)
        converts.append(1.0 if r["rating"] >= 5 else 0.0)
        negs = sampler.sample(seen.get(u, set()), n_neg)
        users.extend([u] * len(negs))
        items.extend(negs)
        labels.extend([0.0] * len(negs))
        converts.extend([0.0] * len(negs))
    return (
        np.asarray(users, dtype=np.int64),
        np.asarray(items, dtype=np.int64),
        np.asarray(labels, dtype=np.float32),
        np.asarray(converts, dtype=np.float32),
    )


@torch.no_grad()
def evaluate(model, data, eval_set, device, kind: str, batch_size: int = 16384):
    model.eval()
    u, it, y, ycv = eval_set
    preds = []
    for i in range(0, len(u), batch_size):
        batch = build_batch(data, u[i : i + batch_size], it[i : i + batch_size], device,
                            with_hist=(kind != "deepfm"))
        out = model(batch)
        if isinstance(out, tuple):
            pctr, pcvr = out
        else:
            pctr, pcvr = out, None
        cols = [pctr.detach().cpu().numpy()]
        if pcvr is not None:
            cols.append(pcvr.detach().cpu().numpy())
            cols.append((pctr * pcvr).detach().cpu().numpy())
        preds.append(np.stack(cols, axis=1))
    model.train()
    P = np.concatenate(preds, axis=0)
    res = {
        "ctr_auc": auc(y, P[:, 0]),
        "ctr_gauc": gauc(y, P[:, 0], u),
        "ctr_logloss": log_loss(y, P[:, 0]),
        "n_eval": int(len(u)),
    }
    if P.shape[1] == 3:
        res["cvr_auc"] = auc(y * ycv, P[:, 1])
        res["ctcvr_auc"] = auc(y * ycv, P[:, 2])
        pos = y * ycv
        res["ctcvr_auc_all"] = auc((pos > 0).astype(float), P[:, 2])
        res["ctcvr_gauc"] = gauc(pos, P[:, 2], u)
    return res, P


def train_one(kind: str, data, args, device, logger) -> dict:
    seed_everything(args.seed)
    users = np.load(data.proc_dir / "rank_users.npy")
    items = np.load(data.proc_dir / "rank_items.npy")
    click = np.load(data.proc_dir / "rank_click.npy").astype(np.float32)
    convert = np.load(data.proc_dir / "rank_convert.npy").astype(np.float32)
    hist_keep = np.load(data.proc_dir / "rank_hist_keep.npy")

    cfg = build_rank_config(data)
    model = MODEL_CLS[kind](cfg).to(device)
    logger.info("[%s] 参数量 %.2fM", kind, model.n_params() / 1e6)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))

    eval_set = make_eval_set(data, args.seed)
    bce = nn.BCELoss()
    best_epoch, best_auc, history = 0, -1.0, []
    steps_total = int(np.ceil(len(users) / args.batch_size)) * args.epochs
    steps = min(steps_total, args.max_steps) if args.max_steps > 0 else steps_total

    with timer(f"训练 {kind}", logger):
        done = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            rng = np.random.default_rng(args.seed + epoch)
            running, nb = 0.0, 0
            for idx in iter_batches(len(users), args.batch_size, device, rng=rng):
                u, it = users[idx], items[idx]
                batch = build_batch(data, u, it, device, with_hist=(kind != "deepfm"),
                                    hist_keep=hist_keep[idx])
                y = torch.as_tensor(click[idx], device=device)
                yc = torch.as_tensor(convert[idx], device=device)
                if kind == "esmm":
                    pctr, pcvr = model(batch)
                    loss = esmm_loss(pctr, pcvr, y, yc)
                else:
                    pred = model(batch)
                    loss = bce(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                running += float(loss.item())
                nb += 1
                done += 1
                if args.max_steps > 0 and done >= args.max_steps:
                    break
            sched.step()
            res, _ = evaluate(model, data, eval_set, device, kind)
            logger.info(
                "[%s] epoch %d loss=%.4f ctr_auc=%.4f gauc=%.4f%s",
                kind, epoch, running / max(nb, 1), res["ctr_auc"], res["ctr_gauc"],
                f" ctcvr_auc={res.get('ctcvr_auc', float('nan')):.4f}" if "ctcvr_auc" in res else "",
            )
            history.append({"epoch": epoch, "loss": running / max(nb, 1), **res})
            key = res["ctr_auc"]
            if key > best_auc:
                best_auc, best_epoch = key, epoch
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "kind": kind},
                           DIR_CKPT / f"rank_{kind}.pt")
            if args.max_steps > 0 and done >= args.max_steps:
                break

    state = torch.load(DIR_CKPT / f"rank_{kind}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    final_res, P = evaluate(model, data, eval_set, device, kind)
    np.save(DIR_RES / f"valid_scores_{kind}.npy", P)
    logger.info("[%s] best epoch=%d %s", kind, best_epoch, json.dumps(final_res, ensure_ascii=False))
    return {"best_epoch": best_epoch, "metrics": final_res, "history": history,
            "params_M": round(model.n_params() / 1e6, 3)}


def main() -> None:
    args = parse_args()
    logger = get_logger("rank_train")
    device = torch.device(args.device) if args.device != "auto" else get_device()
    data = load_processed(Path(args.proc_dir))

    kinds = ["deepfm", "din", "esmm"] if args.model == "all" else [args.model]
    out = {}
    for k in kinds:
        out[k] = train_one(k, data, args, device, logger)
    save_json(out, DIR_RES / "rank_stage.json")

    rows = []
    for k, v in out.items():
        m = v["metrics"]
        rows.append({
            "模型": {"deepfm": "DeepFM", "din": "DIN", "esmm": "ESMM"}[k],
            "参数量(M)": v["params_M"],
            "CTR-AUC": round(m["ctr_auc"], 4),
            "CTR-GAUC": round(m["ctr_gauc"], 4),
            "LogLoss": round(m["ctr_logloss"], 4),
            "CTCVR-AUC": round(m.get("ctcvr_auc", float("nan")), 4),
        })
    print("\n=== 精排 / 多目标评测（valid 集 1正99负）===")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
