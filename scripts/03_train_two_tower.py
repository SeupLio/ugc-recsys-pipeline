#!/usr/bin/env python
"""Step 3：训练双塔召回模型，并导出全库用户 / 物品向量。

关键点：
- 训练目标用 batch 内负样本 softmax + 流行度 logQ 修正，这是工业界召回侧的标准做法；
- 物品塔额外接入 step2 生成的内容语义向量，让「没有任何行为的物品」也能得到可用向量，
  这是后续冷启动召回成立的前提。

用法：
    python scripts/03_train_two_tower.py --epochs 8 --batch_size 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import (  # noqa: E402
    DIR_CKPT,
    DIR_PROC,
    get_device,
    get_logger,
    save_json,
    seed_everything,
    timer,
)
from recsys.data.torch_data import (  # noqa: E402
    build_batch,
    build_item_batch,
    iter_batches,
    load_processed,
)
from recsys.models.two_tower import SampledSoftmaxLoss, TwoTower, TwoTowerConfig  # noqa: E402

DEF = {"epochs": 8, "batch_size": 1024, "lr": 1e-3, "out_dim": 64, "id_dim": 64,
       "tau": 0.05, "dropout": 0.1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--epochs", type=int, default=DEF["epochs"])
    p.add_argument("--batch_size", type=int, default=DEF["batch_size"])
    p.add_argument("--lr", type=float, default=DEF["lr"])
    p.add_argument("--out_dim", type=int, default=DEF["out_dim"])
    p.add_argument("--tau", type=float, default=DEF["tau"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--no_content", action="store_true", help="消融：物品塔去掉语义向量")
    return p.parse_args()


def retrieval_eval(model, data, device, item_counts, eval_pairs, topk=50, batch_size=2048):
    """在 valid 集上粗算 Recall@K（全库暴力检索，扣掉用户训练期已看物品）。"""
    from recsys.eval.metrics import evaluate_topk

    model.eval()
    with torch.no_grad():
        u_vecs, i_vecs = [], []
        users_arr = np.arange(data.num_users)
        for i in range(0, len(users_arr), batch_size):
            u = users_arr[i : i + batch_size]
            b = {
                "user_idx": torch.as_tensor(u, dtype=torch.int64, device=device),
                "gender_idx": data.profile["gender_idx"][u].to(device),
                "age_idx": data.profile["age_idx"][u].to(device),
                "occ_idx": data.profile["occ_idx"][u].to(device),
                "user_dense": data.user_feat[u].to(device),
                "hist_item_idx": data.seq_mat[u].to(device),
            }
            u_vecs.append(model.user_vec(b).cpu())
        full = build_item_batch(data, device)
        for j in range(0, data.num_items, batch_size):
            idx = np.arange(j, min(j + batch_size, data.num_items))
            sub = {k: v[idx] if v.shape[0] == data.num_items else v for k, v in full.items()}
            sub["item_idx"] = torch.as_tensor(idx, dtype=torch.int64, device=device)
            i_vecs.append(model.item_vec(sub).cpu())
    u_vec = torch.cat(u_vecs).numpy()
    i_vec = torch.cat(i_vecs).numpy()

    seen = {}
    for uid, grp in data.interactions[data.interactions["split_tag"] == "train"].groupby("user_idx"):
        seen[int(uid)] = set(grp["item_idx"].tolist())

    u2g, u2p = {}, {}
    eval_u = eval_pairs["user_idx"].unique()
    sims_all = np.ascontiguousarray(u_vec, dtype=np.float32)
    emb_item = np.ascontiguousarray(i_vec, dtype=np.float32)
    for u in eval_u:
        scores = emb_item @ sims_all[int(u)]
        scores[list(seen.get(int(u), set()))] = -np.inf
        cand = np.argpartition(-scores, topk)[:topk]
        pred = cand[np.argsort(-scores[cand])].tolist()
        gt = [int(eval_pairs.loc[eval_pairs["user_idx"] == u, "item_idx"].iloc[0])]
        u2g[int(u)] = gt
        u2p[int(u)] = pred
    return evaluate_topk(u2g, u2p, ks=(20, 50)), u_vec, i_vec


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    logger = get_logger("two_tower")
    device = torch.device(args.device) if args.device != "auto" else get_device()

    data = load_processed(Path(args.proc_dir))
    content_path = Path(args.proc_dir) / "content" / "content_emb.npy"
    item_content = np.load(content_path) if content_path.exists() else None
    if item_content is None and not args.no_content:
        logger.warning("未找到内容向量，退化到无语义输入的双塔")
    logger.info("设备=%s users=%d items=%d", device, data.num_users, data.num_items)

    cfg = TwoTowerConfig(
        num_users=data.num_users,
        num_items=data.num_items,
        num_genres=max(data.num_genres, 1),
        num_gender=int(data.users_df["gender_idx"].max()) + 1,
        num_age=int(data.users_df["age_idx"].max()) + 1,
        num_occ=int(data.users_df["occ_idx"].max()) + 1,
        id_dim=DEF["id_dim"],
        out_dim=args.out_dim,
        dropout=DEF["dropout"],
        use_content=(item_content is not None) and (not args.no_content),
    )
    model = TwoTower(cfg, torch.as_tensor(item_content) if item_content is not None else None).to(device)
    logger.info("参数量: %.2fM", model.n_params() / 1e6)

    train_pos = data.interactions[
        (data.interactions["split_tag"] == "train") & (data.interactions["rating"] >= 4)
    ]
    users = train_pos["user_idx"].to_numpy(dtype=np.int64)
    items = train_pos["item_idx"].to_numpy(dtype=np.int64)
    logger.info("训练正样本: %d", len(users))

    # 因果掩码：召回样本的历史序列同样要剔除「本次曝光之后」的行为，
    # 更不能包含当次正样本本身，否则用户塔会退化成查表。
    counts = train_pos.groupby("user_idx").size()
    keep_arr = np.zeros(len(users), dtype=np.int16)
    pos = 0
    for uid, m in counts.items():
        shown = min(int(m), 50)
        for j in range(int(m)):
            keep_arr[pos + j] = min(max(j - max(0, int(m) - 50), 0), shown)
        pos += int(m)
    assert pos == len(users), (pos, len(users))

    item_pop = data.item_counts / max(data.item_counts.sum(), 1)
    criterion = SampledSoftmaxLoss(torch.as_tensor(item_pop, dtype=torch.float32), tau=args.tau).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    valid_pairs = data.interactions[data.interactions["split_tag"] == "valid"][
        ["user_idx", "item_idx"]
    ]
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        total_loss, nb = 0.0, 0
        for idx in iter_batches(len(users), args.batch_size, device, rng=rng):
            u, it = users[idx], items[idx]
            batch = build_batch(data, u, it, device, with_hist=True,
                                hist_keep=keep_arr[idx])
            u_vec, i_vec = model(batch)
            loss = criterion(u_vec, i_vec, batch["item_idx"], model.logit_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item())
            nb += 1
        sched.step()
        mean_loss = total_loss / max(nb, 1)
        msg = f"epoch {epoch}/{args.epochs} loss={mean_loss:.4f}"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, u_vec, i_vec = retrieval_eval(
                model, data, device, None, valid_pairs, topk=50
            )
            msg += f" | valid recall@20={metrics['recall@20']:.4f} recall@50={metrics['recall@50']:.4f}"
            if metrics["recall@50"] > best:
                best = metrics["recall@50"]
                history.append({"epoch": epoch, **metrics})
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg.__dict__, "metrics": metrics},
                    DIR_CKPT / "two_tower.pt",
                )
        logger.info(msg)

    model.load_state_dict(torch.load(DIR_CKPT / "two_tower.pt", map_location=device)["model"])
    _, u_vec, i_vec = retrieval_eval(model, data, device, None, valid_pairs, topk=50)
    np.save(DIR_PROC / "user_emb.npy", u_vec)
    np.save(DIR_PROC / "item_emb.npy", i_vec)
    save_json(
        {"best_valid_recall@50": float(best), "history": history,
         "use_content": bool(item_content is not None) and not args.no_content,
         "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
        DIR_PROC / "two_tower_train.json",
    )
    logger.info("双塔向量已导出: user=%s item=%s", u_vec.shape, i_vec.shape)


if __name__ == "__main__":
    main()
