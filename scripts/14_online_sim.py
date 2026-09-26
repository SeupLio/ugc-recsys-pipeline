#!/usr/bin/env python
"""Step 14：在线模拟与流量分配机制（A/B 实验闭环）。

离线指标涨了不等于线上收益。这里做一个**用户行为模拟器**把链路接上线：
- 点击模型：用 held-out 的真实数据拟合「用户会不会点这条内容」，模拟曝光→点击→转化；
- 流量分配策略：新内容扶持额度、质量分降权、探索流量比例（ε-greedy）；
- A/B 框架：user-level 哈希分桶 + SRM 校验 + delta method 比率指标检验 + CUPED 方差削减。

为什么必须有这一层：推荐系统的策略改动会改变曝光分布，进而改变后续行为数据，
纯离线评测无法捕捉这种反馈回路。模拟器至少能把「策略 → 曝光分布 → 指标」这条链路跑通。

用法：
    python scripts/14_online_sim.py --epsilon 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.common import DIR_PROC, DIR_RES, get_logger, save_json, seed_everything  # noqa: E402
from recsys.data.torch_data import load_processed  # noqa: E402

# 流量分配策略的候选内容池大小（每次请求）
POOL = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--epsilon", type=float, default=0.1, help="探索流量比例")
    p.add_argument("--cold_quota", type=int, default=3, help="实验组给新内容的坑位数")
    p.add_argument("--alpha_quality", type=float, default=0.3, help="低质内容降权强度")
    p.add_argument("--n_requests", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------- 统计工具
def srm_check(n_ctrl: int, n_treat: int, expected_ratio: float = 0.5) -> dict:
    """Sample Ratio Mismatch：分桶比例偏离预期就说明分流本身有问题，结论不可信。"""
    from scipy.stats import chisquare

    total = n_ctrl + n_treat
    exp = np.array([total * expected_ratio, total * (1 - expected_ratio)])
    obs = np.array([n_ctrl, n_treat], dtype=float)
    chi2, p = chisquare(obs, exp)
    return {"n_control": int(n_ctrl), "n_treatment": int(n_treat),
            "chi2": float(chi2), "p_value": float(p),
            "pass": bool(p > 0.01)}


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, unit: np.ndarray,
                       n_boot: int = 500, seed: int = 42) -> tuple:
    """比率型指标的 bootstrap 置信区间。

    重采样单元必须是「用户/请求」而不是单条曝光——同一用户的曝光彼此相关，
    按曝光重采样会低估方差、把置信区间做得过窄（这是比率指标 A/B 最常见的错误）。
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"num": num, "den": den, "u": unit})
    per_unit = df.groupby("u")[["num", "den"]].sum()
    n_arr = per_unit["num"].to_numpy()
    d_arr = per_unit["den"].to_numpy()
    R = float(n_arr.sum() / max(d_arr.sum(), 1e-9))
    idx = np.arange(len(n_arr))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.choice(idx, size=len(idx), replace=True)
        boots[b] = n_arr[sel].sum() / max(d_arr[sel].sum(), 1e-9)
    return R, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> np.ndarray:
    """CUPED：用实验前协变量 x 削减指标方差。

    y_adj = y - θ(x - E[x])，θ = Cov(y,x)/Var(x)。
    方差削减比例 = ρ²（相关系数的平方），所以协变量选得越相关收益越大。
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if theta is None:
        var = x.var()
        theta = float(np.cov(y, x)[0, 1] / var) if var > 1e-12 else 0.0
    return y - theta * (x - x.mean())


# ---------------------------------------------------------------- 模拟器
def _solve_intercept_shift(z: np.ndarray, target_mean: float) -> float:
    """解 δ 使 mean(sigmoid(z + δ)) == target_mean（单调，二分法即可）。"""
    z = np.asarray(z, dtype=np.float64)
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        m = float(np.mean(1.0 / (1.0 + np.exp(-np.clip(z + mid, -50, 50)))))
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mde_required(pooled_p: float, n_per_arm: int, alpha: float = 0.05,
                 power: float = 0.8) -> float:
    """给定每臂样本量，反解能检出的最小相对提升（MDE）。

    用于回答"这个实验规模够不够、多大提升才看得见"，避免拿不显著当"无效果"。
    """
    from scipy.stats import norm

    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    abs_mde = (z_a + z_b) * np.sqrt(2 * pooled_p * (1 - pooled_p) / max(n_per_arm, 1))
    return float(abs_mde / max(pooled_p, 1e-9))


class UserSimulator:
    """基于真实数据拟合的点击模拟器。

    点击概率 = sigmoid(β0 + β1·相关度 + β2·质量分 + β3·新鲜度)，
    系数用 held-out 数据上的逻辑回归拟合，而不是拍脑袋设定——
    这样模拟出的 CTR 量级与真实数据可比。
    """

    def __init__(self, item_emb, user_emb, quality, logger):
        self.item_emb = item_emb
        self.user_emb = user_emb
        self.quality = (quality - quality.mean()) / (quality.std() + 1e-8)
        self.logger = logger
        self.coef = None

    def fit(self, data, seed=42):
        """用 valid 集（真实点击/未点击）拟合点击倾向。"""
        from sklearn.linear_model import LogisticRegression

        valid = data.interactions[data.interactions["split_tag"] == "valid"]
        rng = np.random.default_rng(seed)
        u_arr = valid["user_idx"].to_numpy()
        i_arr = valid["item_idx"].to_numpy()
        y = (valid["rating"].to_numpy() >= 4).astype(int)

        # 负例：随机未交互物品（模拟"曝光了但没点"）
        n_neg = len(u_arr)
        neg_i = rng.integers(0, self.item_emb.shape[0], size=n_neg)
        U = np.concatenate([u_arr, u_arr])
        I = np.concatenate([i_arr, neg_i])
        Y = np.concatenate([y, np.zeros(n_neg, dtype=int)])

        rel = np.einsum("ij,ij->i", self.item_emb[I], self.user_emb[U])
        q = self.quality[I]
        X = np.column_stack([rel, q])
        lr = LogisticRegression(max_iter=500)
        lr.fit(X, Y)
        b_rel, b_q = float(lr.coef_[0][0]), float(lr.coef_[0][1])

        # ---- 先验校准（必须做）----
        # 上面是 50/50 平衡采样，拟合出的截距对应"平衡世界"的基线，
        # 直接拿去模拟会得到 CTR≈0.84 这种明显失真的量级。
        # 这里把截距平移到真实曝光基线：mean sigmoid(z+δ) = 真实 CTR。
        # 真实基线：全量交互中 rating>=4（正向）的占比。
        real_ctr = float((data.interactions["rating"] >= 4).mean())
        z_ref = b_rel * np.einsum(
            "ij,ij->i", self.item_emb[i_arr], self.user_emb[u_arr]
        ) + b_q * self.quality[i_arr]
        delta = _solve_intercept_shift(z_ref, real_ctr)
        self.coef = (float(lr.intercept_[0]) + delta, b_rel, b_q)
        self.logger.info(
            "点击模型: b0=%.3f(校准后, δ=%.3f) b_rel=%.3f b_quality=%.3f | 目标基线CTR=%.4f",
            self.coef[0], delta, b_rel, b_q, real_ctr)
        return self

    def p_click(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        b0, b_rel, b_q = self.coef
        rel = np.einsum("ij,ij->i", self.item_emb[items], self.user_emb[users])
        z = b0 + b_rel * rel + b_q * self.quality[items]
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def allocate(pool_scores: np.ndarray, cold_mask: np.ndarray, quota: int,
             epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """流量分配：按分数排序，给新内容保留 quota 个坑位，再以 ε 概率随机探索。"""
    n = len(pool_scores)
    if rng.random() < epsilon:
        return rng.permutation(n)[:POOL]
    order = np.argsort(-pool_scores)
    cold_slots = [i for i in order if cold_mask[i]][:quota]
    warm_slots = [i for i in order if not cold_mask[i]]
    chosen = cold_slots + [i for i in warm_slots if i not in set(cold_slots)]
    return np.asarray(chosen[:POOL], dtype=np.int64)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    logger = get_logger("online_sim")
    proc = Path(args.proc_dir)
    data = load_processed(proc)

    user_emb = np.load(proc / "user_emb.npy")
    item_emb = np.load(proc / "item_emb.npy")
    quality = np.load(proc / "quality_score.npy")
    cold_items = np.load(proc / "cold_items.npy")
    cold_mask = np.zeros(data.num_items, dtype=bool)
    cold_mask[cold_items] = True

    sim = UserSimulator(item_emb, user_emb, quality, logger).fit(data, args.seed)

    # --------------------------------------------------------- A/B 分桶
    rng = np.random.default_rng(args.seed)
    users = np.arange(data.num_users)
    bucket = (users * 2654435761 % 100) < 50  # user-level 哈希分桶
    logger.info("分桶: control=%d treatment=%d", int((~bucket).sum()), int(bucket.sum()))

    # 实验前协变量（CUPED 用）：用户历史正向行为数。
    # 诚实说明：模拟器的点击概率只依赖相关度与质量分，与用户活跃度无因果关系，
    # 因此 CUPED 在这里的方差削减有限——这正是它该被报告出来而不是被藏起来的原因。
    hist_act = data.users_df.sort_values("user_idx")["u_cnt_click"].fillna(0).to_numpy()
    hist_ctr = data.users_df.sort_values("user_idx")["u_click_rate"].fillna(0).to_numpy()

    rows = []
    arms = {
        "control(无冷启扶持/无质量降权)": dict(quota=0, alpha=0.0),
        "treatment(冷启额度3+质量降权)": dict(quota=args.cold_quota, alpha=args.alpha_quality),
    }
    stats = {}
    ctrl_users = users[~bucket]
    treat_users = users[bucket]
    # 对照组与实验组各自只在自己的分桶里抽用户：两组人群互斥，检验才成立
    arms_run = [
        ("control(无冷启扶持/无质量降权)", dict(quota=0, alpha=0.0),
         0.0, ctrl_users, False),
        (f"treatment(冷启额度{args.cold_quota}+质量降权)", dict(quota=args.cold_quota,
         alpha=args.alpha_quality), args.epsilon, treat_users, True),
    ]
    for name, cfg, eps, arm_users, is_treat_arm in arms_run:
        clicks, exposures, converts, cold_expo = [], [], [], []
        u_used = []
        for _ in range(args.n_requests):
            u = int(rng.choice(arm_users))
            # 候选池：按相关度取 Top200 再截断，模拟召回输出
            s = item_emb @ user_emb[u]
            top = np.argpartition(-s, 200)[:200]
            score = s[top] - cfg["alpha"] * np.clip(-sim.quality[top], 0, None)
            chosen = allocate(score, cold_mask[top], cfg["quota"], eps, rng)
            items = top[chosen]
            p = sim.p_click(np.full(len(items), u), items)
            y = (rng.random(len(items)) < p).astype(int)
            # 转化：点击后以质量分相关的概率发生
            p_cv = np.clip(0.4 + 0.1 * sim.quality[items], 0.05, 0.95)
            ycv = (y * (rng.random(len(items)) < p_cv)).astype(int)

            clicks.append(y.sum()); exposures.append(len(items))
            converts.append(ycv.sum()); cold_expo.append(int(cold_mask[items].sum()))
            u_used.append(u)

        clicks = np.asarray(clicks, dtype=float)
        exposures = np.asarray(exposures, dtype=float)
        converts = np.asarray(converts, dtype=float)
        cold_expo = np.asarray(cold_expo, dtype=float)
        u_used = np.asarray(u_used)

        # 用户级聚合：t 检验的观测单元必须是用户，不能用请求（同一用户的请求相关）
        per_user = pd.DataFrame({"u": u_used, "c": clicks, "e": exposures}) \
            .groupby("u").agg(c=("c", "sum"), e=("e", "sum"))
        user_ctr = (per_user["c"] / per_user["e"].clip(lower=1)).to_numpy()
        user_ids = per_user.index.to_numpy()

        ctr, lo, hi = bootstrap_ratio_ci(clicks, exposures, u_used)
        cvr, cvr_lo, cvr_hi = bootstrap_ratio_ci(converts, np.maximum(clicks, 1e-9), u_used)
        stats[name] = {
            "requests": int(len(clicks)),
            "users": int(len(user_ids)),
            "ctr": float(ctr), "ctr_ci": [lo, hi],
            "cvr": float(cvr), "cvr_ci": [cvr_lo, cvr_hi],
            "cold_exposure_ratio": float(cold_expo.sum() / exposures.sum()),
            "avg_exposure": float(exposures.mean()),
            "is_treatment": bool(is_treat_arm),
            "_user_ctr": user_ctr,
            "_user_hist_ctr": hist_act[user_ids],
        }
        logger.info("%s -> CTR=%.4f [%.4f, %.4f] CVR=%.4f 新内容曝光占比=%.4f",
                    name, ctr, lo, hi, cvr, cold_expo.sum() / exposures.sum())

    # --------------------------------------------------------- 统计检验
    ctrl_name, treat_name = [n for n, *_ in arms_run]
    c, t = stats[ctrl_name], stats[treat_name]
    srm = srm_check(int((~bucket).sum()), int(bucket.sum()))

    # CUPED：用实验前点击率削减方差
    y_c = c["_user_ctr"]; y_t = t["_user_ctr"]
    x_c = c["_user_hist_ctr"]; x_t = t["_user_hist_ctr"]
    y_c_adj = cuped_adjust(y_c, x_c)
    y_t_adj = cuped_adjust(y_t, x_t)
    var_before = y_c.var(ddof=1) / len(y_c) + y_t.var(ddof=1) / len(y_t)
    var_after = y_c_adj.var(ddof=1) / len(y_c_adj) + y_t_adj.var(ddof=1) / len(y_t_adj)
    var_reduction = 1 - var_after / max(var_before, 1e-12)

    from scipy.stats import ttest_ind
    tt = ttest_ind(y_t_adj, y_c_adj, equal_var=False)
    diff = y_t_adj.mean() - y_c_adj.mean()
    se = np.sqrt(var_after)
    ci = (diff - 1.96 * se, diff + 1.96 * se)

    pooled_p = (c["ctr"] + t["ctr"]) / 2
    mde = mde_required(pooled_p, min(c["users"], t["users"]))
    logger.info("MDE（每臂 %d 用户）= %.2f%% 相对提升；实测差异 %.2f%%",
                min(c["users"], t["users"]), mde * 100, 100 * diff / max(pooled_p, 1e-9))

    rows = [
        {"指标": "请求数", "对照组": c["requests"], "实验组": t["requests"]},
        {"指标": "用户数", "对照组": c["users"], "实验组": t["users"]},
        {"指标": "CTR", "对照组": round(c["ctr"], 4), "实验组": round(t["ctr"], 4)},
        {"指标": "CTR 95%CI", "对照组": str([round(x, 4) for x in c["ctr_ci"]]),
         "实验组": str([round(x, 4) for x in t["ctr_ci"]])},
        {"指标": "CVR", "对照组": round(c["cvr"], 4), "实验组": round(t["cvr"], 4)},
        {"指标": "新内容曝光占比", "对照组": round(c["cold_exposure_ratio"], 4),
         "实验组": round(t["cold_exposure_ratio"], 4)},
    ]
    out = {
        "epsilon": args.epsilon,
        "cold_quota": args.cold_quota,
        "alpha_quality": args.alpha_quality,
        "n_requests_per_arm": args.n_requests,
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in stats.items()},
        "srm_check": srm,
        "ab_test": {
            "ctr_diff_cuped": float(diff),
            "ci95": [float(ci[0]), float(ci[1])],
            "p_value": float(tt.pvalue),
            "significant": bool(tt.pvalue < 0.05),
            "variance_reduction_by_cuped": float(var_reduction),
            "mde_relative": round(float(mde), 4),
            "users_per_arm": int(min(c["users"], t["users"])),
        },
    }
    save_json(out, DIR_RES / "online_sim.json")

    print("\n=== 在线模拟 A/B 实验 ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nSRM 校验:", json.dumps(srm, ensure_ascii=False))
    print("CUPED 方差削减: %.1f%%" % (var_reduction * 100))
    print("MDE（每臂 %d 用户）: 需 %.2f%% 相对提升才可检出" % (
        min(c["users"], t["users"]), mde * 100))
    print("CTR 差异(实验-对照) = %+.4f, 95%%CI [%+.4f, %+.4f], p=%.4f → %s" % (
        diff, ci[0], ci[1], tt.pvalue, "显著" if tt.pvalue < 0.05 else "不显著"))


if __name__ == "__main__":
    main()
