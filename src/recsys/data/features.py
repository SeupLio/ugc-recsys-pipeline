"""用户 / 物品画像统计与训练-验证-测试切分。

原则：
1. 画像统计只用 train 段数据 —— 画像里掺入 valid/test 的行为就是标准的未来信息泄漏，
   离线指标会虚高（本仓库 docs/实验结果说明.md 记录的第二次泄漏事故即来源于此）；
2. leave-one-out 切分：每个用户按时间排序，最后一条 → test，倒数第二条 → valid，
   其余 → train。预测的是「下一个」，而不是「随机某一条」；
3. 每个统计都提供 DuckDB SQL 与 pandas 两条实现（use_sql 切换），
   SQL 路径用于展示工业界的特征工程写法，pandas 路径作为零依赖兜底。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

try:
    import duckdb  # type: ignore

    HAS_DUCKDB = True
except Exception:  # pragma: no cover
    HAS_DUCKDB = False


def _duckdb_split(inter: pd.DataFrame) -> pd.DataFrame:
    rel = duckdb.sql(
        """
        SELECT user_idx, item_idx,
               CASE WHEN rn = 1 AND rating >= 4 THEN 'test'
                    WHEN rn = 2 AND rating >= 4 THEN 'valid'
                    ELSE 'train' END AS split_tag
        FROM (
            SELECT user_idx, item_idx, rating,
                   row_number() OVER (
                       PARTITION BY user_idx ORDER BY ts DESC, item_idx DESC
                   ) AS rn
            FROM inter
        )
        """
    )
    return rel.df()


def assign_split(inter: pd.DataFrame, use_sql: bool = True) -> pd.DataFrame:
    """留一法切分。注意：只有「正向行为」才有资格做 test/valid 的 ground truth，
    低分交互（rating<4）全部归 train（它们是曝光上下文，不是预测目标）。"""
    if use_sql and HAS_DUCKDB:
        return _duckdb_split(inter)
    df = inter.sort_values(["user_idx", "ts", "item_idx"], kind="mergesort")
    grp = df.groupby("user_idx", sort=False)
    rn = grp.cumcount()
    total = grp["user_idx"].transform("size")
    rn_from_tail = total - 1 - rn  # 0 = 最后一条
    tags = np.where(
        (rn_from_tail == 0) & (df["rating"].to_numpy() >= 4), "test",
        np.where((rn_from_tail == 1) & (df["rating"].to_numpy() >= 4), "valid", "train"),
    )
    return pd.DataFrame({
        "user_idx": df["user_idx"].to_numpy(),
        "item_idx": df["item_idx"].to_numpy(),
        "split_tag": tags,
    })


def user_features(train_only: pd.DataFrame, use_sql: bool = True) -> pd.DataFrame:
    """用户画像（仅 train 统计）。列名与 users.csv 落盘一致。"""
    if use_sql and HAS_DUCKDB:
        return _duckdb_user_features(train_only)
    df = train_only.copy()
    df["dt_month"] = pd.to_datetime(df["ts"], unit="s").dt.to_period("M")
    g = df.groupby("user_idx")
    out = pd.DataFrame({
        "u_cnt_total": g.size(),
        "u_cnt_click": g.apply(lambda x: int((x["rating"] >= 4).sum())),
        "u_cnt_convert": g.apply(lambda x: int((x["rating"] >= 5).sum())),
        "u_mean_rating": g["rating"].mean(),
        "u_std_rating": g["rating"].std(ddof=0),
        "u_convert_rate": g.apply(
            lambda x: (x["rating"] >= 5).sum() / max((x["rating"] >= 4).sum(), 1)),
        "u_click_rate": g.apply(
            lambda x: (x["rating"] >= 4).sum() / max(len(x), 1)),
        "u_active_months": g["dt_month"].nunique(),
    })
    # 最近 10 条评分均值：先按时间排，再取每组尾部
    ordered = df.sort_values(["user_idx", "ts"], kind="mergesort")
    out["u_recent10_mean_rating"] = ordered.groupby("user_idx")["rating"].apply(
        lambda s: s.tail(10).mean())
    out = out.reset_index()
    out["u_std_rating"] = out["u_std_rating"].fillna(0.0)
    return out


def _duckdb_user_features(train_only: pd.DataFrame) -> pd.DataFrame:
    rel = duckdb.sql(
        """
        SELECT user_idx,
               count(*) AS u_cnt_total,
               sum(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) AS u_cnt_click,
               sum(CASE WHEN rating = 5 THEN 1 ELSE 0 END) AS u_cnt_convert,
               avg(rating) AS u_mean_rating,
               coalesce(stddev_pop(rating), 0) AS u_std_rating,
               sum(CASE WHEN rating = 5 THEN 1 ELSE 0 END)
                   / greatest(sum(CASE WHEN rating >= 4 THEN 1 ELSE 0 END), 1) AS u_convert_rate,
               sum(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) / greatest(count(*), 1) AS u_click_rate,
               count(DISTINCT ts / 2629800) AS u_active_months,
               avg(recent10) AS u_recent10_mean_rating
        FROM (
            SELECT *, CASE WHEN rn_asc >= total - 9 THEN rating END AS recent10
            FROM (
                SELECT user_idx, rating, ts,
                       row_number() OVER (PARTITION BY user_idx ORDER BY ts ASC, item_idx ASC) AS rn_asc,
                       count(*) OVER (PARTITION BY user_idx) AS total
                FROM train_only
            )
        )
        GROUP BY user_idx ORDER BY user_idx
        """
    )
    return rel.df()


def item_features(train_only: pd.DataFrame, use_sql: bool = True,
                  half_life_days: float = 180.0) -> pd.DataFrame:
    """物品画像（仅 train 统计）。i_score_decay 为时间衰减热度。"""
    if use_sql and HAS_DUCKDB:
        return _duckdb_item_features(train_only, half_life_days)
    df = train_only.copy()
    g = df.groupby("item_idx")
    out = pd.DataFrame({
        "i_cnt_total": g.size(),
        "i_cnt_click": g.apply(lambda x: int((x["rating"] >= 4).sum())),
        "i_cnt_convert": g.apply(lambda x: int((x["rating"] >= 5).sum())),
        "i_n_unique_users": g["user_idx"].nunique(),
        "i_mean_rating": g["rating"].mean(),
        "i_convert_rate": g.apply(
            lambda x: (x["rating"] >= 5).sum() / max((x["rating"] >= 4).sum(), 1)),
    })
    max_ts = float(df["ts"].max())
    age_days = (max_ts - df["ts"].to_numpy(dtype=np.float64)) / 86400.0
    df = df.assign(_decay=np.exp(-age_days / float(half_life_days)))
    out["i_score_decay"] = df.groupby("item_idx")["_decay"].sum()
    out = out.reset_index()
    return out


def _duckdb_item_features(train_only: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    rel = duckdb.sql(
        f"""
        WITH s AS (
            SELECT item_idx, rating, user_idx, ts,
                   max(ts) OVER () AS max_ts
            FROM train_only
        )
        SELECT item_idx,
               count(*) AS i_cnt_total,
               sum(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) AS i_cnt_click,
               sum(CASE WHEN rating = 5 THEN 1 ELSE 0 END) AS i_cnt_convert,
               count(DISTINCT user_idx) AS i_n_unique_users,
               avg(rating) AS i_mean_rating,
               sum(CASE WHEN rating = 5 THEN 1 ELSE 0 END)
                   / greatest(sum(CASE WHEN rating >= 4 THEN 1 ELSE 0 END), 1) AS i_convert_rate,
               sum(exp(-(max_ts - ts) / 86400.0 / {float(half_life_days)})) AS i_score_decay
        FROM s GROUP BY item_idx ORDER BY item_idx
        """
    )
    return rel.df()
