#!/usr/bin/env python
"""Step 19：pCTR 校准评测 —— AUC 之外的「绝对值正确性」。

流程：加载 4 个精排模型检查点（deepfm / din / esmm / mmoe），在
验证集（1 正 + 99 负）上打分，然后：

1. 校准诊断（校准前）：ECE / PCOC / 可靠性曲线；
2. 用验证集的**一半**拟合 PAV 保序回归，**另一半**上报校准后指标
   （防止「用测试集拟合校准器」的泄漏）；
3. 结论表：校准前后 ECE / PCOC 对比。

面试可讲：AUC（序）与校准（绝对值）是两个正交的维度——排序模型
完全可以 AUC 0.75 但整体高估 3 倍；计费 / 竞价 / E&E 探索预算全都
依赖后者。

用法：python scripts/19_calibrate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from recsys.common import DIR_CKPT, DIR_PROC, DIR_RES, get_logger, save_json  # noqa: E402
from recsys.data.torch_data import build_batch, load_processed  # noqa: E402
from recsys.models.calibration import (  # noqa: E402
    ece, isotonic_apply, isotonic_fit, pcoc, reliability_bins)
from recsys.models.multitask import MMoERanker  # noqa: E402
from recsys.models.rank import DeepFMRanker, DINRanker, ESMM, RankConfig  # noqa: E402

# 复用 05 的评测集构造（1 正 + 99 流行度负，与 AUC 口径一致）
_spec = importlib.util.spec_from_file_location(
    "tr05", ROOT / "scripts" / "05_train_rank.py")
_05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_05)
make_eval_set = _05.make_eval_set

log = get_logger("calibrate")
CLS = {"deepfm": DeepFMRanker, "din": DINRanker, "esmm": ESMM, "mmoe": MMoERanker}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", type=str, default="deepfm,din,esmm,mmoe")
    return p.parse_args()


@torch.no_grad()
def score_all(model, data, u, it, kind: str) -> np.ndarray:
    """批量打分：多任务模型取 pCTR（校准对象是 CTR 概率）。"""
    model.eval()
    preds = []
    for i in range(0, len(u), 16384):
        batch = build_batch(data, u[i: i + 16384], it[i: i + 16384],
                            torch.device("cpu"), with_hist=(kind != "deepfm"))
        out = model(batch)
        pctr = out[0] if isinstance(out, (tuple, list)) else out
        preds.append(pctr.numpy())
    return np.concatenate(preds)


def main() -> None:
    args = parse_args()
    data = load_processed(DIR_PROC)
    u, it, y, _yc = make_eval_set(data, args.seed)

    cfg = RankConfig(
        num_users=data.num_users, num_items=data.num_items,
        num_genres=max(data.num_genres, 1),
        num_gender=int(data.users_df["gender_idx"].max()) + 1,
        num_age=int(data.users_df["age_idx"].max()) + 1,
        num_occ=int(data.users_df["occ_idx"].max()) + 1)

    # 校准拟合/评测各用一半（同一分布内切分，防止泄漏）
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(u))
    fit_idx, eval_idx = perm[: len(perm) // 2], perm[len(perm) // 2:]

    out = {}
    for kind in args.models.split(","):
        ckpt = DIR_CKPT / f"rank_{kind}.pt"
        if not ckpt.exists():
            log.warning("跳过 %s（无检查点 %s）", kind, ckpt)
            continue
        model = CLS[kind](cfg)
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
        s = score_all(model, data, u, it, kind).astype(np.float64)

        s_fit, y_fit = s[fit_idx], y[fit_idx]
        s_ev, y_ev = s[eval_idx], y[eval_idx]

        table = isotonic_fit(s_fit, y_fit)
        s_ev_cal = isotonic_apply(s_ev, table)

        row = {
            "ece_before": round(ece(s_ev, y_ev), 5),
            "ece_after": round(ece(s_ev_cal, y_ev), 5),
            "pcoc_before": round(pcoc(s_ev, y_ev), 3),
            "pcoc_after": round(pcoc(s_ev_cal, y_ev), 3),
            "mean_pred": round(float(s_ev.mean()), 5),
            "mean_label": round(float(y_ev.mean()), 5),
            "cal_table_points": len(table),
            "reliability_before": [
                [round(c, 3), round(o, 4), n]
                for c, o, n in reliability_bins(s_ev, y_ev)],
        }
        out[kind] = row
        log.info("%-7s ECE %.5f → %.5f（%+.0f%%）PCOC %.2f → %.2f "
                 "（均值分 %.4f vs 真实 %.4f）",
                 kind, row["ece_before"], row["ece_after"],
                 (row["ece_after"] / max(row["ece_before"], 1e-9) - 1) * 100,
                 row["pcoc_before"], row["pcoc_after"],
                 row["mean_pred"], row["mean_label"])

    save_json(out, DIR_RES / "calibration.json")
    log.info("要点：负采样训练的排序模型天然高估（PCOC >> 1 是常态而非 bug）；"
             "保序回归把 ECE 压到接近 0，同时保持 AUC 不变（单调映射不改序）")


if __name__ == "__main__":
    main()
