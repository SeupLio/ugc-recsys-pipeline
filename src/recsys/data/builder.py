"""序列 / padding 构建工具与流行度加权负采样器。"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set

import numpy as np
import pandas as pd

PAD_ID = -1


def pad_seq(items: Sequence[int], max_len: int) -> np.ndarray:
    """行为序列右对齐 padding（PAD=-1）：只保留最近 max_len 条。

    右对齐的原因：离当前时刻越近的行为对预测「下一次」越重要，
    截断时丢掉的是最老的头部，而不是最近的尾部。
    """
    seq = [int(x) for x in items][-max_len:]
    n_pad = max_len - len(seq)
    return np.array([PAD_ID] * n_pad + seq, dtype=np.int64)


def pad_genres(ids: Sequence[int], max_genres: int) -> tuple[np.ndarray, np.ndarray]:
    """类目 id 列表 padding（0 补齐）+ mask（1=真实类目，0=padding）。"""
    seq = [int(x) for x in ids][:max_genres]
    mask = np.zeros(max_genres, dtype=np.int64)
    mask[: len(seq)] = 1
    padded = np.zeros(max_genres, dtype=np.int64)
    padded[: len(seq)] = seq
    return padded, mask


def build_sequences(train_pos: pd.DataFrame, max_len: int) -> Dict[int, List[int]]:
    """按时间顺序构建每个用户的行为序列（右侧为最新），截到最近 max_len 条。

    train_pos 需包含 user_idx / item_idx / ts 三列。
    """
    df = train_pos.sort_values(["user_idx", "ts"], kind="mergesort")
    seqs: Dict[int, List[int]] = defaultdict(list)
    for uid, iid in zip(df["user_idx"].to_numpy(), df["item_idx"].to_numpy()):
        seqs[int(uid)].append(int(iid))
    return {uid: lst[-max_len:] for uid, lst in seqs.items()}


class NegativeSampler:
    """按 popularity^beta 加权（有放回）的负采样器。

    beta=0.75 是「曝光偏差」折中：纯均匀采样会让大量从未曝光的长尾物品
    被当作强负信号；纯按流行度又会被头部垄断。0.75 次幂让负样本分布
    接近真实曝光分布、同时保留中长尾（见 docs/系统设计.md §1）。

    - sample(seen, n):      抽 n 条不在 seen 中的负例（池子小时允许重复）
    - sample_batch(users, seen): 逐用户各抽 1 条，规避该用户已看物品，向量化实现
    """

    def __init__(self, num_items: int, counts: np.ndarray, beta: float = 0.75, seed: int = 42):
        self.num_items = int(num_items)
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape[0] != self.num_items:
            raise ValueError(f"counts 长度 {counts.shape[0]} != num_items {self.num_items}")
        w = np.power(np.maximum(counts, 0.0), float(beta))
        if w.sum() <= 0:
            w = np.ones(self.num_items)
        self.probs = (w / w.sum()).astype(np.float64)
        self.rng = np.random.default_rng(seed)

    def _one_avoiding(self, seen: Set[int]) -> int:
        """拒绝采样抽 1 条不在 seen 的物品；池子耗尽时退化为均匀补集。"""
        for _ in range(64):
            cand = int(self.rng.choice(self.num_items, p=self.probs))
            if cand not in seen:
                return cand
        rest = np.flatnonzero(np.ones(self.num_items, dtype=bool))
        rest = np.array([i for i in rest if i not in seen], dtype=np.int64)
        if len(rest) == 0:
            return int(self.rng.integers(0, self.num_items))
        return int(self.rng.choice(rest))

    def sample(self, seen: Set[int], n: int) -> np.ndarray:
        seen = set(seen)
        draws = self.rng.choice(self.num_items, size=n, p=self.probs)
        for i in range(n):
            if draws[i] in seen:
                draws[i] = self._one_avoiding(seen)
        return draws.astype(np.int64)

    def sample_batch(self, users: np.ndarray, seen: Dict[int, Set[int]]) -> np.ndarray:
        users = np.asarray(users)
        draws = self.rng.choice(self.num_items, size=len(users), p=self.probs)
        # 命中已看物品的位置逐个重抽（实际占比很小）
        for i in range(len(users)):
            s = seen.get(int(users[i]))
            if s and draws[i] in s:
                draws[i] = self._one_avoiding(s)
        return draws.astype(np.int64)
