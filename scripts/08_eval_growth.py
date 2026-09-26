#!/usr/bin/env python
"""Step 8：用户增长 —— 相似人群扩展（Lookalike）与用户价值预估（pLTV）。

为什么这里不用 step1 的留一法切分：
留一法下每个用户未来只有 1 条行为，「用户价值」退化成 0/1 且几乎人人命中，
指标会假得毫无意义。增长类任务必须按**时间窗口**切：
- past window（ts < 80% 分位）只用来算特征与行为向量；
- future window（ts >= 分位点）只用来定义价值标签并做验证。

为避免先导未来信息，用户向量由 past window 的行为矩阵经 TruncatedSVD 现算，
不复用在整个训练集上学出来的双塔向量（那会让指标虚高）。

用法：
    python scripts/08_eval_growth.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import DIR_PROC, DIR_RES, get_logger, save_json, seed_everything  # noqa: E402
from recsys.data.torch_data import load_processed  # noqa: E402
from recsys.eval.metrics import auc, lift_at_k, rmse, spearman  # noqa: E402
from recsys.growth.lookalike import Lookalike, UserValueModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--future_ratio", type=float, default=0.2, help="未来窗口占时间轴比例")
    p.add_argument("--seed_ratio", type=float, default=0.02, help="种子用户占全量比例")
    p.add_argument("--expand_ratio", type=float, default=0.10, help="扩充人群占候选池比例")
    p.add_argument("--topk_per_seed", type=int, default=20)
    p.add_argument("--svd_dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    logger = get_logger("growth")
    proc = Path(args.proc_dir)
    data = load_processed(proc)

    inter = data.interactions[data.interactions["split_tag"].isin(["train", "valid", "test"])]
    T = float(np.quantile(inter["ts"].to_numpy(), 1 - args.future_ratio))
    past = inter[inter["ts"] < T]
    future = inter[inter["ts"] >= T]
    pos_past = past[past["rating"] >= 4]
    pos_future = future[future["rating"] >= 4]
    logger.info("时间切分点 ts=%.0f | past=%d future=%d", T, len(past), len(future))

    num_users, num_items = data.num_users, data.num_items
    past_cnt = np.zeros(num_users)
    future_cnt = np.zeros(num_users)
    np.add.at(past_cnt, pos_past["user_idx"].to_numpy(), 1.0)
    np.add.at(future_cnt, pos_future["user_idx"].to_numpy(), 1.0)

    # 只有在过去窗口有行为、且未来窗口可被观测的用户才参与评测
    eligible = np.where(past_cnt >= 3)[0]
    base_rate_future = future_cnt[eligible].mean()
    thr = np.quantile(future_cnt[eligible], 0.8)
    y_high = (future_cnt >= thr).astype(float)
    logger.info("候选用户 %d，未来平均正向行为 %.2f，高价值阈值 %.1f，正例占比 %.3f",
                len(eligible), base_rate_future, thr, y_high[eligible].mean())

    # ----------------------------------------------- 过去窗口 → SVD 用户/物品向量
    rows = pos_past["user_idx"].to_numpy()
    cols = pos_past["item_idx"].to_numpy()
    mat = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_users, num_items),
                     dtype=np.float32)
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
    U = svd.fit_transform(mat)
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
    logger.info("SVD 解释方差 %.3f", float(svd.explained_variance_ratio_.sum()))

    # ------------------------------------------------------------- pLTV 模型
    user = data.users_df.sort_values("user_idx")
    past_feat = np.column_stack([
        np.log1p(past_cnt),
        np.log1p(past.groupby("user_idx").size().reindex(range(num_users), fill_value=0).to_numpy()),
        past.groupby("user_idx")["rating"].mean().reindex(range(num_users), fill_value=0).to_numpy(),
        user["u_convert_rate"].fillna(0).to_numpy(),
        user["u_click_rate"].fillna(0).to_numpy(),
        user["u_active_months"].fillna(0).to_numpy(),
        user["u_recent10_mean_rating"].fillna(user["u_mean_rating"]).to_numpy(),
    ])
    X = np.concatenate([past_feat, U], axis=1)
    names = ["log_past_pos", "log_past_total", "past_mean_rating", "convert_rate",
             "click_rate", "active_months", "recent10_mean"] + [f"svd_{i}" for i in range(U.shape[1])]

    idx = eligible
    y_cont = future_cnt          # 与 user_idx 对齐的全长向量
    y_bin = y_high
    tr_idx, te_idx = train_test_split(idx, test_size=0.3, random_state=args.seed)
    vm = UserValueModel().fit(X[tr_idx], y_cont[tr_idx], names)
    pred = vm.predict(X[te_idx])
    m_rmse = rmse(y_cont[te_idx], pred)
    m_corr = spearman(pred, y_cont[te_idx])
    m_auc = float(roc_auc_score(y_bin[te_idx], pred)) if len(set(y_bin[te_idx])) > 1 else float("nan")
    m_lift = lift_at_k(y_bin[te_idx], pred, 0.1)
    importance = vm.top_feature_importance(X[te_idx], y_cont[te_idx], k=5)
    logger.info("pLTV: RMSE=%.3f Spearman=%.4f HighValue-AUC=%.4f Lift@10%%=%.2f",
                m_rmse, m_corr, m_auc, m_lift)

    # ------------------------------------------------------------- Lookalike
    # 种子 = 过去窗口里最活跃的一小撮已知用户（模拟「已沉淀的高价值名单」）
    order_past = np.argsort(-past_cnt)
    n_seed = max(10, int(len(eligible) * args.seed_ratio))
    seed_users = np.array([u for u in order_past if u in set(eligible.tolist())][:n_seed])
    pool = np.setdiff1d(eligible, seed_users)
    logger.info("种子用户 %d，候选池 %d", len(seed_users), len(pool))

    look = Lookalike(U)
    rows_out = []

    def report(name: str, exp: np.ndarray) -> None:
        hit = float(y_high[exp].mean())
        rows_out.append({
            "人群包": name,
            "人群规模": int(len(exp)),
            "高价值占比": round(hit, 4),
            "相对大盘Lift": round(hit / max(base_rate_high, 1e-8), 3),
            "人均未来正向行为": round(float(future_cnt[exp].mean()), 3),
        })

    base_rate_high = float(y_high[pool].mean())
    report("质心扩展(centroid)", np.asarray(
        look.expand_centroid(seed_users, pool, ratio=args.expand_ratio)))
    knn = np.asarray(look.expand_knn(seed_users, pool, topk_per_seed=args.topk_per_seed,
                                     ratio=args.expand_ratio))
    report(f"KNN投票扩展(top{args.topk_per_seed}/种子)", knn)
    rng = np.random.default_rng(args.seed)
    report("随机采样(基线)", rng.choice(pool, size=max(1, int(len(pool) * args.expand_ratio)),
                                        replace=False))
    # 两步法：先按相似度扩人群，再用 pLTV 筛掉低价值用户
    score = vm.predict(X[knn])
    keep = knn[np.argsort(-score)[: max(1, len(knn) // 2)]]
    report("KNN扩展 + pLTV再筛选(Top50%)", keep)

    out = {
        "future_ratio": args.future_ratio,
        "cutoff_ts": float(T),
        "num_eligible_users": int(len(eligible)),
        "base_high_value_rate_in_pool": round(base_rate_high, 4),
        "avg_future_positive": round(float(base_rate_future), 4),
        "svd_explained_var": round(float(svd.explained_variance_ratio_.sum()), 4),
        "pltv": {
            "rmse": round(float(m_rmse), 4), "spearman": round(float(m_corr), 4),
            "high_value_auc": round(float(m_auc), 4), "lift_at_10pct": round(float(m_lift), 3),
            "top_features": [[n, round(float(v), 5)] for n, v in importance],
        },
        "lookalike": rows_out,
    }
    save_json(out, DIR_RES / "growth_stage.json")
    print("\n=== Lookalike 人群包（候选池基准高价值占比 %.4f）===" % base_rate_high)
    print(pd.DataFrame(rows_out).to_string(index=False))
    print("\n=== pLTV 价值模型（预测未来窗口正向行为数）===")
    print(pd.DataFrame([{
        "RMSE": round(float(m_rmse), 4), "Spearman": round(float(m_corr), 4),
        "高价值AUC": round(float(m_auc), 4), "Lift@10%": round(float(m_lift), 3),
    }]).to_string(index=False))
    print("Top 特征:", importance)


if __name__ == "__main__":
    main()
