#!/usr/bin/env python
"""Step 11：内容质量评估 + 用户意图画像（内容理解的应用层）。

对应「质量评估及策略分析」「用户意图理解、内容与素材理解、标签生成」：

1) **内容质量分**：用贝叶斯平滑的 CTR/CVR 做后验（小样本内容不会因为偶然的高点击率
   被高估），再与内容语义侧的「可发现性」组合成可入库的 quality score。
   质量分用于：召回加权、冷启动的先验排序、以及低质内容的过滤。

2) **用户意图画像**：把用户历史行为映射到受控标签词表上的分布，得到一个可解释的
   意图向量。它能直接支撑标签倒排召回，也能给策略分析提供「用户在找什么」的口径。

用法：
    python scripts/11_content_quality.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import DIR_PROC, DIR_RES, get_logger, save_json  # noqa: E402
from recsys.data.torch_data import load_processed  # noqa: E402
from recsys.eval.metrics import spearman  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--prior_m", type=float, default=50.0, help="贝叶斯平滑的先验强度（等效曝光数）")
    return p.parse_args()


def bayesian_smooth_rate(numer: np.ndarray, denom: np.ndarray, prior: float, m: float) -> np.ndarray:
    """贝叶斯平滑比率：(Σx + m·prior) / (Σn + m)。

    低曝光内容的原始比率方差极大（1 次曝光 1 次点击 = CTR 100%），直接用会让
    质量分被噪声主导。m 是先验的「等效样本量」，控制向全局均值收缩的强度。
    """
    numer = np.asarray(numer, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)
    return (numer + m * prior) / (denom + m)


def gini(values: np.ndarray) -> float:
    """基尼系数：0=完全均匀，1=完全集中。用于量化流量集中度。"""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def main() -> None:
    args = parse_args()
    logger = get_logger("quality")
    proc = Path(args.proc_dir)
    data = load_processed(proc)

    tags = json.loads((proc / "content" / "content_tags.json").read_text(encoding="utf-8"))
    content = np.load(proc / "content" / "content_emb.npy")

    # ------------------------------------------------------------ 曝光/点击/转化
    # 关键：split_tag=='train' 只包含正向行为，低分交互被标为 'unclick'。
    # 若只用 train 当曝光，CTR 会恒等于 1（分母只含点击）。
    # 正确口径：曝光 = 时间窗前 80% 的全部交互（含未点击）；后 20% 留作 held-out 验证。
    # 不用留一法的 test/valid 时间戳定界——那是每个用户各自的最后一次行为，
    # 全局最小值会落在时间轴很早的位置，切出来的"未来窗口"几乎等于全量。
    all_ts = data.interactions["ts"].to_numpy(dtype=np.float64)
    t_cut = float(np.quantile(all_ts, 0.8))
    window = data.interactions[data.interactions["ts"] < t_cut]
    logger.info("曝光窗口: ts < %.0f，共 %d 条交互（含未点击）", t_cut, len(window))
    num_items = data.num_items

    expo = window.groupby("item_idx").size().reindex(range(num_items), fill_value=0).to_numpy()
    click = window[window["rating"] >= 4].groupby("item_idx").size().reindex(
        range(num_items), fill_value=0).to_numpy()
    conv = window[window["rating"] >= 5].groupby("item_idx").size().reindex(
        range(num_items), fill_value=0).to_numpy()

    prior_ctr = click.sum() / max(expo.sum(), 1)
    prior_cvr = conv.sum() / max(click.sum(), 1)
    ctr_s = bayesian_smooth_rate(click, expo, prior_ctr, args.prior_m)
    cvr_s = bayesian_smooth_rate(conv, click, prior_cvr, args.prior_m)
    logger.info("全局先验 CTR=%.4f CVR=%.4f，平滑强度 m=%.0f", prior_ctr, prior_cvr, args.prior_m)

    # -------------------------------------------- 内容侧「可发现性」：语义近邻密度
    # 与全库平均相似度高的内容更容易被语义召回捞到；低者需要靠协同信号。
    # 用抽样近似 O(N^2)，6k 抽样足以稳定估计。
    rng = np.random.default_rng(0)
    sub = rng.choice(num_items, size=min(1500, num_items), replace=False)
    sim = content @ content[sub].T  # (N, 1500)
    discoverability = sim.mean(axis=1)

    # ------------------------------------------------------------ 质量分合成
    def z(x):
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean()) / (x.std() + 1e-8)

    quality = 0.55 * z(ctr_s) + 0.25 * z(cvr_s) + 0.20 * z(discoverability)
    # 曝光过少时质量分不可信，收缩到 0（中性）
    low_conf = expo < 5
    quality = np.where(low_conf, 0.0, quality)
    logger.info("质量分: mean=%.4f std=%.4f，低置信内容 %d 条被置中性",
                quality.mean(), quality.std(), int(low_conf.sum()))

    # ------------------------------------------------- 验证：质量分是否有预测力
    # Held-out 检验：用未来窗口（ts >= t_cut）的每一条真实交互，看质量分能否把
    # 「会被点击的曝光」排在「不会被点击的曝光」之前。这比按物品算 CTR 再相关更严格，
    # 因为留一法下每个物品在 valid/test 里只有 1 条曝光，物品级 CTR 无法估计。
    from recsys.eval.metrics import auc as auc_fn

    future = data.interactions[data.interactions["ts"] >= t_cut]
    y = (future["rating"] >= 4).astype(float).to_numpy()
    s_q = quality[future["item_idx"].to_numpy()]
    s_raw = (click / np.maximum(expo, 1))[future["item_idx"].to_numpy()]
    s_pop = np.log1p(expo)[future["item_idx"].to_numpy()]
    auc_q = auc_fn(y, s_q)
    auc_raw = auc_fn(y, s_raw)
    auc_pop = auc_fn(y, s_pop)
    rho = spearman(quality, click / np.maximum(expo, 1))
    logger.info("未来窗口 %d 条交互：质量分 AUC=%.4f（原始CTR=%.4f，热度=%.4f）",
                len(future), auc_q, auc_raw, auc_pop)

    # ------------------------------------------------------------ 用户意图画像
    inv_user: Dict = {}
    tag_names = sorted({t for v in tags.values() for t in v})
    tag_idx = {t: i for i, t in enumerate(tag_names)}
    intent = np.zeros((data.num_users, len(tag_names)), dtype=np.float32)
    for u in range(data.num_users):
        hist = [int(i) for i in data.seq_mat[u][data.seq_mat[u] >= 0].numpy()]
        if not hist:
            continue
        w = np.zeros(len(tag_names), dtype=np.float64)
        # 时间衰减权重：越近的行为越能代表当下意图
        for j, it in enumerate(hist):
            decay = 1.0 / np.log2(len(hist) - j + 1)
            for t in tags.get(str(it), []):
                w[tag_idx[t]] += decay
        s = w.sum()
        if s > 0:
            intent[u] = (w / s).astype(np.float32)
    top_tags = np.argsort(-intent, axis=1)[:, :3]
    active = intent.sum(axis=1) > 0
    logger.info("意图画像: %d 个用户有标签分布，平均熵=%.3f",
                int(active.sum()),
                float(np.mean([-(p * np.log(p + 1e-9)).sum()
                               for p in intent[active]])))

    # --------------------------------------------------------------- 落盘
    np.save(proc / "quality_score.npy", quality)
    np.save(proc / "intent_matrix.npy", intent)
    save_json({"tag_names": tag_names}, proc / "intent_tags.json")

    df_q = pd.DataFrame({
        "item_idx": np.arange(num_items),
        "exposure": expo, "click": click, "convert": conv,
        "ctr_smooth": ctr_s, "cvr_smooth": cvr_s,
        "discoverability": discoverability, "quality": quality,
    })
    df_q = df_q.merge(data.items_df[["item_idx", "clean_title", "genres"]], on="item_idx", how="left")
    df_q.to_csv(DIR_RES / "content_quality.csv", index=False)

    stat = {
        "prior_ctr": round(float(prior_ctr), 4),
        "prior_cvr": round(float(prior_cvr), 4),
        "smoothing_m": args.prior_m,
        "future_interactions": int(len(future)),
        "quality_auc_future_click": round(float(auc_q), 4),
        "raw_ctr_auc_future_click": round(float(auc_raw), 4),
        "popularity_auc_future_click": round(float(auc_pop), 4),
        "num_items_scored": int((~low_conf).sum()),
        "num_users_with_intent": int(active.sum()),
        "intent_tag_vocab": len(tag_names),
        "exposure_gini": round(gini(expo), 4),
    }
    save_json(stat, DIR_RES / "quality_stage.json")

    print("\n=== 内容质量评估 ===")
    print(pd.DataFrame([stat]).T.to_string(header=False))
    print("\n=== 质量分 Top10 ===")
    print(df_q.nlargest(10, "quality")[["clean_title", "exposure", "ctr_smooth", "quality"]]
          .to_string(index=False))
    print("\n=== 质量分 Bottom5（曝光充足但质量低）===")
    print(df_q[df_q["exposure"] >= 50].nsmallest(5, "quality")[
        ["clean_title", "exposure", "ctr_smooth", "quality"]].to_string(index=False))
    print("\n=== 用户意图画像示例（用户 7）===")
    u = 7
    top = [(tag_names[i], round(float(intent[u][i]), 4)) for i in top_tags[u]]
    print(top)


if __name__ == "__main__":
    main()
