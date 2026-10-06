#!/usr/bin/env python
"""Step 18：MMoE 多任务精排 —— 与 ESMM 的同输入结构对照。

ESMM 用「共享子网络串联」耦合 CTR/CVR，MMoE 用「并行专家 + 任务门」
解耦（工业多任务 → PLE 的基础）。同 embedding、同标签、同 esmm_loss，
唯一变量是结构，差距可干净归因于门控。

评测三口径（与 05 完全一致）：
- ctr_auc  / cvr_auc / ctcvr_auc（点击空间）
- gate JS 散度（两任务门是否学出分化）

用法：python scripts/18_train_mmoe.py [--epochs 3] [--experts 4]
"""

from __future__ import annotations

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch
import torch.nn as nn  # noqa: E402

from recsys.common import (  # noqa: E402
    DIR_CKPT, DIR_PROC, DIR_RES, get_logger, save_json, seed_everything, timer)
from recsys.data.torch_data import build_batch, iter_batches, load_processed  # noqa: E402
from recsys.models.multitask import MMoERanker  # noqa: E402
from recsys.models.rank import RankConfig, esmm_loss  # noqa: E402

# 复用 05 的评测基建（同一份评测集与三口径指标，保证与 ESMM 严格可比）
_spec = importlib.util.spec_from_file_location(
    "tr05", ROOT / "scripts" / "05_train_rank.py")
_05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_05)
make_eval_set, evaluate = _05.make_eval_set, _05.evaluate

log = get_logger("train_mmoe")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--experts", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cpu")
    data = load_processed(DIR_PROC)

    users = np.load(DIR_PROC / "rank_users.npy")
    items = np.load(DIR_PROC / "rank_items.npy")
    click = np.load(DIR_PROC / "rank_click.npy").astype(np.float32)
    convert = np.load(DIR_PROC / "rank_convert.npy").astype(np.float32)

    cfg = RankConfig(
        num_users=data.num_users, num_items=data.num_items,
        num_genres=max(data.num_genres, 1),
        num_gender=int(data.users_df["gender_idx"].max()) + 1,
        num_age=int(data.users_df["age_idx"].max()) + 1,
        num_occ=int(data.users_df["occ_idx"].max()) + 1)
    model = MMoERanker(cfg, n_experts=args.experts).to(device)
    log.info("[mmoe] 参数量 %.2fM（%d 专家 ×2 门 ×2 塔）",
             model.n_params() / 1e6, args.experts)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    eval_set = make_eval_set(data, args.seed)
    bce = nn.BCELoss()

    best_auc, best_epoch, history = -1.0, 0, []
    steps_total = int(np.ceil(len(users) / args.batch_size)) * args.epochs
    steps = min(steps_total, args.max_steps) if args.max_steps > 0 else steps_total
    done = 0
    with timer("训练 mmoe", log):
        for epoch in range(1, args.epochs + 1):
            model.train()
            rng = np.random.default_rng(args.seed + epoch)
            running, nb = 0.0, 0
            for idx in iter_batches(len(users), args.batch_size, device, rng=rng):
                u, it = users[idx], items[idx]
                batch = build_batch(data, u, it, device, with_hist=False)
                y = torch.as_tensor(click[idx], device=device)
                yc = torch.as_tensor(convert[idx], device=device)
                pctr, pcvr = model(batch)
                loss = esmm_loss(pctr, pcvr, y, yc)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                running, nb = running + float(loss), nb + 1
                done += 1
                if args.max_steps > 0 and done >= args.max_steps:
                    break
            sched.step()
            res, _P = evaluate(model, data, eval_set, device, "esmm")
            key = res["ctcvr_auc"]
            history.append({"epoch": epoch, "loss": round(running / max(nb, 1), 4),
                            **{k: round(v, 4) for k, v in res.items()
                               if isinstance(v, (int, float))}})
            log.info("epoch %d loss=%.4f ctr_auc=%.4f cvr_auc=%.4f ctcvr_auc=%.4f",
                     epoch, running / max(nb, 1), res["ctr_auc"],
                     res["cvr_auc"], res["ctcvr_auc"])
            if key > best_auc:
                best_auc, best_epoch = key, epoch
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "kind": "mmoe"}, DIR_CKPT / "rank_mmoe.pt")
            if args.max_steps > 0 and done >= args.max_steps:
                break

    # 门分化诊断：两任务门在验证集上的平均 JS 散度
    u, it = eval_set[0], eval_set[1]
    js = model.gate_divergence(
        build_batch(data, u[:4096], it[:4096], device, with_hist=False))

    out = {"model": "mmoe", "experts": args.experts, "epochs": args.epochs,
           "best_epoch": best_epoch, "gate_js_divergence": round(js, 5),
           "history": history}
    save_json(out, DIR_RES / "mmoe.json")
    log.info("完成：best ctcvr_auc=%.4f（epoch %d），门 JS 散度 %.5f%s",
             best_auc, best_epoch, js,
             "（门已分化，任务间存在张力）" if js > 0.02 else "（门趋同，任务相关性高）")


if __name__ == "__main__":
    main()
