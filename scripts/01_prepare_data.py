#!/usr/bin/env python
"""Step 1：数据准备。

产出：
- data/processed/vocab.json        词表
- data/processed/users.csv         用户画像（含 DuckDB SQL 计算的行为统计）
- data/processed/items.csv         物品画像（含类目、衰减热度）
- data/processed/interactions.csv  全量交互 + split_tag(train/valid/test/cold)
- data/processed/sequences.npz     DIN 行为序列矩阵 (num_users, max_len)
- data/processed/rank_*.npy        排序模型训练样本（正负样本已预生成，保证可复现）
- data/processed/cold_*.npy        冷启动物品及独立评测集

用法：
    python scripts/01_prepare_data.py --neg_ratio 4 --cold_frac 0.03
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

from recsys.common import DIR_PROC, DIR_RAW, save_json, seed_everything, timer  # noqa: E402
from recsys.data import builder, features  # noqa: E402
from recsys.data.dataset import (  # noqa: E402
    CLICK_THRESHOLD,
    CONVERT_THRESHOLD,
    build_frames,
    build_vocab,
    load_raw,
)

logger = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MovieLens-1M -> 推荐系统训练样本")
    p.add_argument("--raw_dir", type=str, default=str(DIR_RAW))
    p.add_argument("--out_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--neg_ratio", type=int, default=4, help="每个正样本配几条负样本")
    p.add_argument("--cold_frac", type=float, default=0.03, help="模拟新内容的物品比例")
    p.add_argument("--max_seq_len", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_sql", action="store_true", help="不使用 DuckDB，走 pandas 兜底")
    return p.parse_args()


def main() -> None:
    global logger
    args = parse_args()
    seed_everything(args.seed)
    from recsys.common import get_logger

    logger = get_logger("prepare")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. 读取
    with timer("读取原始数据", logger):
        ratings, movies, users = load_raw(Path(args.raw_dir))
        vocab = build_vocab(ratings, users, movies)
        frames = build_frames(ratings, movies, users, vocab)
        inter = frames["interactions"]
        items = frames["items"]
        users_df = frames["users"]

    num_users, num_items = vocab.num_users, vocab.num_items
    logger.info("规模: users=%d items=%d interactions=%d", num_users, num_items, len(inter))

    # ------------------------------------------------- 2. 模拟新内容（冷启动）
    rng = np.random.default_rng(args.seed)
    pos = inter[inter["rating"] >= CLICK_THRESHOLD].copy()
    item_pos_cnt = pos.groupby("item_idx").size().reindex(range(num_items), fill_value=0)
    eligible = np.where(item_pos_cnt.to_numpy() >= 5)[0]
    n_cold = max(1, int(len(eligible) * args.cold_frac))
    cold_items = np.sort(rng.choice(eligible, size=n_cold, replace=False))
    logger.info("冷启动物品数: %d (占比 %.2f%%)", len(cold_items), 100.0 * len(cold_items) / num_items)

    cold_mask = inter["item_idx"].isin(cold_items).to_numpy()
    cold_pairs = inter[cold_mask][["user_idx", "item_idx", "rating", "ts"]].copy()
    cold_pairs = cold_pairs[cold_pairs["rating"] >= CLICK_THRESHOLD]
    inter_visible = inter[~cold_mask].copy()  # 训练可见部分：冷启动物品全部抹除

    # --------------------------------------------------------- 3. 特征 & 切分
    use_sql = not args.no_sql
    with timer("留一法切分", logger):
        split_df = features.assign_split(inter_visible, use_sql=use_sql)

    inter_visible = inter_visible.merge(
        split_df[["user_idx", "item_idx", "split_tag"]], on=["user_idx", "item_idx"], how="left"
    )
    inter_visible["split_tag"] = inter_visible["split_tag"].fillna("unclick")

    use_sql = not args.no_sql
    # 只用训练集统计画像：画像里一旦掺入 valid/test 的行为，就是标准的未来信息泄漏，
    # 会让离线指标虚高、线上 A/B 直接打脸。
    with timer("用户/物品画像特征(仅 train 统计)", logger):
        train_only = inter_visible[inter_visible["split_tag"] == "train"]
        uf = features.user_features(train_only, use_sql=use_sql)
        itf = features.item_features(train_only, use_sql=use_sql)

    # 冷启动评测集：从未参与训练的物品-用户对中取正向行为作为 ground truth
    cold_pairs["split_tag"] = "cold"
    full = pd.concat([inter_visible, cold_pairs], ignore_index=True)

    users_df = users_df.merge(uf, on="user_idx", how="left")
    items = items.merge(itf, on="item_idx", how="left")
    for col in ["u_cnt_total", "u_cnt_click", "u_cnt_convert"]:
        users_df[col] = users_df[col].fillna(0).astype(int)
    for col in ["i_cnt_total", "i_cnt_click", "i_cnt_convert", "i_n_unique_users"]:
        items[col] = items[col].fillna(0).astype(int)
    for col in ["u_mean_rating", "u_std_rating", "u_recent10_mean_rating",
                "u_convert_rate", "u_click_rate"]:
        users_df[col] = users_df[col].fillna(0.0)
    for col in ["i_mean_rating", "i_convert_rate", "i_score_decay"]:
        items[col] = items[col].fillna(0.0)

    # ------------------------------------------------------------ 4. 行为序列
    train_pos = full[(full["split_tag"] == "train") & (full["rating"] >= CLICK_THRESHOLD)]
    seq = builder.build_sequences(train_pos, max_len=args.max_seq_len)
    seq_mat = np.full((num_users, args.max_seq_len), -1, dtype=np.int32)
    for uid, items_list in seq.items():
        arr = np.asarray(items_list, dtype=np.int32)
        seq_mat[uid, args.max_seq_len - len(arr) :] = arr

    # ------------------------------------------------------------ 5. 排序样本
    train_all = inter_visible[inter_visible["split_tag"] == "train"]
    pos_pairs = train_all[train_all["rating"] >= CLICK_THRESHOLD][["user_idx", "item_idx", "rating"]]
    if not pos_pairs["user_idx"].is_monotonic_increasing:
        pos_pairs = pos_pairs.sort_values(["user_idx", "ts"] if "ts" in pos_pairs.columns
                                          else ["user_idx"]).reset_index(drop=True)
    item_counts = train_all.groupby("item_idx").size().reindex(range(num_items), fill_value=0).to_numpy()
    sampler = builder.NegativeSampler(num_items, item_counts, beta=0.75, seed=args.seed)

    seen: dict[int, set[int]] = {}
    for uid, grp in train_pos.groupby("user_idx"):
        seen[int(uid)] = set(grp["item_idx"].tolist())
    # 训练窗口内所有交互（含低分）也要当曝光处理，避免给模型送已看过的负样本
    for uid, grp in train_all.groupby("user_idx"):
        seen.setdefault(int(uid), set()).update(grp["item_idx"].tolist())

    r = args.neg_ratio
    n = len(pos_pairs)
    sample_users = np.zeros(n * (r + 1), dtype=np.int32)
    sample_items = np.zeros(n * (r + 1), dtype=np.int32)
    sample_click = np.zeros(n * (r + 1), dtype=np.int8)
    sample_convert = np.zeros(n * (r + 1), dtype=np.int8)

    u_arr = pos_pairs["user_idx"].to_numpy()
    i_arr = pos_pairs["item_idx"].to_numpy()
    rating_arr = pos_pairs["rating"].to_numpy()
    sample_users[:: (r + 1)] = u_arr
    sample_items[:: (r + 1)] = i_arr
    sample_click[:: (r + 1)] = 1
    sample_convert[:: (r + 1)] = (rating_arr >= CONVERT_THRESHOLD).astype(np.int8)

    logger.info("开始负采样: 正样本 %d × %d 负样本", n, r)
    for offset in range(1, r + 1):
        negs = sampler.sample_batch(u_arr, seen)
        sample_users[offset :: (r + 1)] = u_arr
        sample_items[offset :: (r + 1)] = negs
    logger.info("样本总量: %d", len(sample_users))

    # 因果掩码：序列里只保留「曝光时刻之前」的行为，否则 DIN 类模型会把
    # 目标物品本身当作证据，形成标签泄漏。
    hist_keep = np.zeros(len(sample_users), dtype=np.int16)
    counts_per_user = pos_pairs.groupby("user_idx").size()
    offset = 0
    for uid, m in counts_per_user.items():
        shown = min(int(m), args.max_seq_len)  # 序列右对齐后实际保留的条数
        for j in range(int(m)):
            k = j - max(0, int(m) - args.max_seq_len)
            k = min(max(k, 0), shown)
            base_row = (offset + j) * (r + 1)
            hist_keep[base_row : base_row + r + 1] = k
        offset += int(m)
    np.save(out / "rank_hist_keep.npy", hist_keep)
    logger.info("因果掩码生成完毕，样例: %s", hist_keep[: (r + 1) * 3].tolist())

    # ---------------------------------------------------------------- 6. 落盘
    vocab.save(out / "vocab.json")
    items.to_csv(out / "items.csv", index=False)
    users_df.to_csv(out / "users.csv", index=False)
    full.to_csv(out / "interactions.csv", index=False)
    np.savez_compressed(out / "sequences.npz", seq_mat=seq_mat)
    np.save(out / "rank_users.npy", sample_users)
    np.save(out / "rank_items.npy", sample_items)
    np.save(out / "rank_click.npy", sample_click)
    np.save(out / "rank_convert.npy", sample_convert)
    np.save(out / "cold_items.npy", cold_items.astype(np.int32))

    test_df = full[full["split_tag"] == "test"][["user_idx", "item_idx"]]
    valid_df = full[full["split_tag"] == "valid"][["user_idx", "item_idx"]]
    test_df.to_csv(out / "test_pairs.csv", index=False)
    valid_df.to_csv(out / "valid_pairs.csv", index=False)
    cold_pairs.to_csv(out / "cold_pairs.csv", index=False)

    item_cnt_vec = np.zeros(num_items, dtype=np.float64)
    cnt = train_all.groupby("item_idx").size()
    item_cnt_vec[cnt.index.to_numpy()] = cnt.to_numpy(dtype=np.float64)
    np.save(out / "item_counts.npy", item_cnt_vec)

    stat = {
        "num_users": int(num_users),
        "num_items": int(num_items),
        "num_genres": int(vocab.num_genres),
        "num_interactions": int(len(inter)),
        "num_cold_items": int(len(cold_items)),
        "num_cold_pairs": int(len(cold_pairs)),
        "train_pos": int(n),
        "rank_samples": int(len(sample_users)),
        "neg_ratio": int(r),
        "max_seq_len": int(args.max_seq_len),
        "test_users": int(test_df["user_idx"].nunique()),
        "cold_test_users": int(cold_pairs["user_idx"].nunique()),
        "click_threshold": CLICK_THRESHOLD,
        "convert_threshold": CONVERT_THRESHOLD,
    }
    save_json(stat, out / "dataset_stat.json")
    logger.info("数据集落盘完成: %s", out)
    print(json.dumps(stat, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
