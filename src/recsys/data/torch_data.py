"""processed 落盘文件 → 统一的 ProcessedData 对象 → PyTorch 训练 batch。

设计要点：
- 所有画像统计特征在 load 时做 log1p + z-score 标准化（仅用全量统计，不含标签信息），
  避免不同量纲（次数 ~1e3 vs 比率 ~1e-2）让 MLP 学不动；
- build_batch 是唯一batch 组装入口：召回（双塔）、精排（DeepFM/DIN/ESMM）、
  对照实验（12_hard_negative）共用同一份特征口径，保证「同输入对照」成立；
- causal_truncate 实现 DIN 的因果掩码：序列里只保留目标曝光之前的 k 条行为。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import pandas as pd
import torch

from .builder import pad_genres, pad_seq

USER_NUM_COLS = [
    "u_cnt_total", "u_cnt_click", "u_cnt_convert", "u_mean_rating", "u_std_rating",
    "u_convert_rate", "u_click_rate", "u_recent10_mean_rating", "u_active_months",
]
USER_LOG_COLS = ["u_cnt_total", "u_cnt_click", "u_cnt_convert"]
ITEM_NUM_COLS = [
    "i_cnt_total", "i_cnt_click", "i_cnt_convert", "i_n_unique_users",
    "i_mean_rating", "i_convert_rate", "i_score_decay",
]
ITEM_LOG_COLS = ["i_cnt_total", "i_cnt_click", "i_cnt_convert", "i_n_unique_users", "i_score_decay"]
MAX_GENRES = 3


def _standardize(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    return ((mat - mu) / np.maximum(sd, 1e-8)).astype(np.float32)


@dataclass
class ProcessedData:
    """一次加载，全链路共享。"""

    proc_dir: Path
    vocab: Dict
    num_users: int
    num_items: int
    num_genres: int
    users_df: pd.DataFrame
    items_df: pd.DataFrame
    interactions: pd.DataFrame
    item_counts: np.ndarray
    seq_mat: torch.Tensor                     # (num_users, max_seq_len)，右对齐，-1=padding
    profile: Dict[str, torch.Tensor]          # gender_idx / age_idx / occ_idx（按 user_idx 索引）
    user_feat: torch.Tensor                   # (num_users, F_u) 已标准化
    item_feat: torch.Tensor                   # (num_items, F_i) 已标准化
    genre_ids: torch.Tensor                   # (num_items, MAX_GENRES)，0=padding
    genre_mask: torch.Tensor                  # (num_items, MAX_GENRES)，1=真实类目
    hist_keep: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    rank_users: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    rank_items: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    rank_click: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    rank_convert: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    valid_pairs: Optional[pd.DataFrame] = None
    cold_items: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    @property
    def max_seq_len(self) -> int:
        return int(self.seq_mat.shape[1])

    @property
    def user_dense_dim(self) -> int:
        return int(self.user_feat.shape[1])

    @property
    def item_dense_dim(self) -> int:
        return int(self.item_feat.shape[1])


def load_processed(proc_dir: Path) -> ProcessedData:
    proc_dir = Path(proc_dir)
    vocab = json.loads((proc_dir / "vocab.json").read_text(encoding="utf-8"))
    num_users, num_items = int(vocab["num_users"]), int(vocab["num_items"])
    num_genres = int(vocab["num_genres"])

    users_df = pd.read_csv(proc_dir / "users.csv").sort_values("user_idx").reset_index(drop=True)
    items_df = pd.read_csv(proc_dir / "items.csv").sort_values("item_idx").reset_index(drop=True)
    interactions = pd.read_csv(proc_dir / "interactions.csv")

    item_counts = np.load(proc_dir / "item_counts.npy")
    seq_mat = torch.as_tensor(np.load(proc_dir / "sequences.npz")["seq_mat"], dtype=torch.int64)

    # 用户侧稠密特征
    users_full = users_df.set_index("user_idx").reindex(range(num_users))
    u_mat = users_full[USER_NUM_COLS].fillna(0.0).to_numpy(dtype=np.float64)
    for c in USER_LOG_COLS:
        u_mat[:, USER_NUM_COLS.index(c)] = np.log1p(np.maximum(u_mat[:, USER_NUM_COLS.index(c)], 0.0))
    user_feat = torch.as_tensor(_standardize(u_mat))

    # 物品侧稠密特征（items.csv 只含全库物品；统计列缺失补 0）
    items_full = items_df.set_index("item_idx")
    miss_cols = [c for c in ITEM_NUM_COLS if c not in items_full.columns]
    for c in miss_cols:
        items_full[c] = 0.0
    i_mat = items_full[ITEM_NUM_COLS].reindex(range(num_items)).fillna(0.0).to_numpy(dtype=np.float64)
    for c in ITEM_LOG_COLS:
        i_mat[:, ITEM_NUM_COLS.index(c)] = np.log1p(np.maximum(i_mat[:, ITEM_NUM_COLS.index(c)], 0.0))
    item_feat = torch.as_tensor(_standardize(i_mat))

    # 类目多值特征
    gi, gm = [], []
    for _, row in items_full.reset_index().sort_values("item_idx").iterrows():
        ids, mask = pad_genres(json.loads(str(row.get("genre_ids", "[]"))), MAX_GENRES)
        gi.append(ids)
        gm.append(mask)
    genre_ids = torch.as_tensor(np.stack(gi) if gi else np.zeros((num_items, MAX_GENRES), dtype=np.int64))
    genre_mask = torch.as_tensor(np.stack(gm) if gm else np.zeros((num_items, MAX_GENRES), dtype=np.int64))

    # 口径细节：items_df 只有出现过交互的物品才落过盘 —— 全量物品以词表为准
    if len(genre_ids) < num_items:
        pad_n = num_items - len(genre_ids)
        genre_ids = torch.cat([genre_ids, torch.zeros(pad_n, MAX_GENRES, dtype=torch.int64)])
        genre_mask = torch.cat([genre_mask, torch.zeros(pad_n, MAX_GENRES, dtype=torch.int64)])

    profile = {
        "gender_idx": torch.as_tensor(users_df["gender_idx"].to_numpy(), dtype=torch.int64),
        "age_idx": torch.as_tensor(users_df["age_idx"].to_numpy(), dtype=torch.int64),
        "occ_idx": torch.as_tensor(users_df["occ_idx"].to_numpy(), dtype=torch.int64),
    }

    def _opt(name: str) -> np.ndarray:
        p = proc_dir / name
        return np.load(p) if p.exists() else None

    hist_keep = _opt("rank_hist_keep.npy")
    rank_users, rank_items = _opt("rank_users.npy"), _opt("rank_items.npy")
    rank_click, rank_convert = _opt("rank_click.npy"), _opt("rank_convert.npy")
    cold_items = _opt("cold_items.npy")
    valid_pairs = (
        pd.read_csv(proc_dir / "valid_pairs.csv") if (proc_dir / "valid_pairs.csv").exists() else None
    )

    return ProcessedData(
        proc_dir=proc_dir, vocab=vocab, num_users=num_users, num_items=num_items,
        num_genres=num_genres, users_df=users_df, items_df=items_df,
        interactions=interactions, item_counts=np.asarray(item_counts, dtype=np.float64),
        seq_mat=seq_mat, profile=profile, user_feat=user_feat, item_feat=item_feat,
        genre_ids=genre_ids, genre_mask=genre_mask,
        hist_keep=hist_keep if hist_keep is not None else np.zeros(0, dtype=np.int16),
        rank_users=rank_users if rank_users is not None else np.zeros(0, dtype=np.int32),
        rank_items=rank_items if rank_items is not None else np.zeros(0, dtype=np.int32),
        rank_click=rank_click if rank_click is not None else np.zeros(0, dtype=np.int8),
        rank_convert=rank_convert if rank_convert is not None else np.zeros(0, dtype=np.int8),
        valid_pairs=valid_pairs,
        cold_items=cold_items if cold_items is not None else np.zeros(0, dtype=np.int32),
    )


def causal_truncate(hist: torch.Tensor, keep: np.ndarray) -> torch.Tensor:
    """因果掩码：每行只保留前 k 条真实行为（k=keep[i]），其余置 -1。

    「前 k 条」指从左到右数的前 k 个非 padding 位置 —— 序列右对齐时，
    越靠右越新，保留前 k 条等价于把「目标物品曝光之后才发生的行为」丢掉，
    这是修复 DIN 标签泄漏的关键（tests/test_data.py 有护栏测试）。
    """
    hist = torch.as_tensor(hist, dtype=torch.int64)
    keep = np.asarray(keep).astype(np.int64)
    valid = (hist >= 0)
    cum = valid.cumsum(dim=1)  # 第 i 个位置是第几条真实行为（1-based）
    allow = valid & (cum <= torch.as_tensor(keep, device=hist.device).unsqueeze(1))
    return torch.where(allow, hist, torch.full_like(hist, -1))


def build_batch(
    data: ProcessedData,
    users: np.ndarray,
    items: np.ndarray,
    device=None,
    with_hist: bool = True,
    hist_keep: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """组装训练/推理 batch。users / items 为等长下标数组。

    with_hist=False 时不带 hist_item_idx（DeepFM 等不看行为的模型），
    hist_keep 传入时先做因果截断（DIN 防标签泄漏）。
    """
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    dev = device if device is not None else torch.device("cpu")

    def to(t: torch.Tensor) -> torch.Tensor:
        return t.to(dev, non_blocking=True)

    batch = {
        "user_idx": to(torch.as_tensor(users, dtype=torch.int64)),
        "item_idx": to(torch.as_tensor(items, dtype=torch.int64)),
        "gender_idx": to(data.profile["gender_idx"][users]),
        "age_idx": to(data.profile["age_idx"][users]),
        "occ_idx": to(data.profile["occ_idx"][users]),
        "user_dense": to(data.user_feat[users]),
        "item_dense": to(data.item_feat[items]),
        "genre_ids": to(data.genre_ids[items]),
        "genre_mask": to(data.genre_mask[items]),
    }
    if with_hist:
        hist = data.seq_mat[users]
        if hist_keep is not None:
            hist = causal_truncate(hist, np.asarray(hist_keep))
        batch["hist_item_idx"] = to(hist)
    return batch


def build_item_batch(data: ProcessedData, device=None) -> Dict[str, torch.Tensor]:
    """全库物品 batch（导出 item 向量 / 全库打分用）。"""
    dev = device if device is not None else torch.device("cpu")
    idx = torch.arange(data.num_items, dtype=torch.int64)
    return {
        "item_idx": idx.to(dev),
        "item_dense": data.item_feat.to(dev),
        "genre_ids": data.genre_ids.to(dev),
        "genre_mask": data.genre_mask.to(dev),
    }


def iter_batches(
    n: int, batch_size: int, device=None, rng: Optional[np.random.Generator] = None
) -> Iterator[torch.Tensor]:
    """随机打乱后按 batch_size 切片，yield LongTensor 下标。

    注意：yield 的是 CPU 张量 —— 调用方常用它去索引 numpy 样本数组
    （users[idx] / items[idx]），CUDA 张量无法做 numpy 下标。
    """
    order = np.arange(n)
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, n, batch_size):
        yield torch.as_tensor(order[i : i + batch_size], dtype=torch.int64)
