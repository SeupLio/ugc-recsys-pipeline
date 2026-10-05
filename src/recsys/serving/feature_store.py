"""在线特征存储（Feature Store，教学级实现）。

生产系统里特征有「两副面孔」：训练时离线批表，服务时在线 KV。
两者不一致（训练-服务偏差 training-serving skew）是推荐系统最经典的
线上掉点原因之一。本模块用「快照 + 读缓存 + 新鲜度追踪」模拟在线侧：

- Snapshot：一次性物化的特征视图（num_users×F / num_items×F），
  带版本号与生成时间；
- FeatureStore：线程安全读取 + LRU 热点缓存 + TTL 新鲜度检查，
  `get_user(uid)` / `get_item(iid)` 返回 (向量, 版本, 是否过期)；
- 若快照超龄（max_staleness）标记 stale，上层可选择降级（换缓存/兜底榜）。

不做 Redis 依赖：接口按 KV 语义设计，换后端只需替换 `_load_snapshot`。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class Snapshot:
    """一次物化的特征视图。version 单调递增。"""

    version: int
    created_at: float
    user_feat: np.ndarray          # (num_users, F_u)
    item_feat: np.ndarray          # (num_items, F_i)
    item_counts: np.ndarray        # (num_items,) 流行度，兜底通道用

    @property
    def age_s(self) -> float:
        return time.time() - self.created_at


class FeatureStore:
    """线程安全的在线特征读取口。

    参数
    ----
    user_feat / item_feat / item_counts : 初始快照内容
    max_staleness : 快照最大允许年龄（秒），超龄 get_* 返回 stale=True
    cache_size    : 每类键的 LRU 容量（用户/物品分别独立 LRU）
    """

    def __init__(self, user_feat: np.ndarray, item_feat: np.ndarray,
                 item_counts: np.ndarray, *, max_staleness: float = 3600.0,
                 cache_size: int = 512, clock=time.time) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._max_staleness = float(max_staleness)
        self._snap = Snapshot(version=1, created_at=clock(),
                              user_feat=np.ascontiguousarray(user_feat),
                              item_feat=np.ascontiguousarray(item_feat),
                              item_counts=np.asarray(item_counts))
        self._u_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._i_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cache_size = int(cache_size)
        self.stats = {"user_hits": 0, "user_misses": 0,
                      "item_hits": 0, "item_misses": 0}

    # ---------------- 对外接口 ----------------

    def get_user(self, uid: int) -> Tuple[Optional[np.ndarray], int, bool]:
        """返回 (特征向量, 快照版本, 是否过期)。越界返回 (None, v, False)。"""
        with self._lock:
            if not (0 <= uid < self._snap.user_feat.shape[0]):
                return None, self._snap.version, False
            vec = self._u_cache.get(uid)
            if vec is not None:
                self._u_cache.move_to_end(uid)
                self.stats["user_hits"] += 1
            else:
                vec = self._snap.user_feat[uid]
                self._u_cache[uid] = vec
                self._evict(self._u_cache)
                self.stats["user_misses"] += 1
            return vec, self._snap.version, self.is_stale()

    def get_item(self, iid: int) -> Tuple[Optional[np.ndarray], int, bool]:
        with self._lock:
            if not (0 <= iid < self._snap.item_feat.shape[0]):
                return None, self._snap.version, False
            vec = self._i_cache.get(iid)
            if vec is not None:
                self._i_cache.move_to_end(iid)
                self.stats["item_hits"] += 1
            else:
                vec = self._snap.item_feat[iid]
                self._i_cache[iid] = vec
                self._evict(self._i_cache)
                self.stats["item_misses"] += 1
            return vec, self._snap.version, self.is_stale()

    def popularity(self, iid: int) -> int:
        """兜底热度（快照内的物品计数）。"""
        with self._lock:
            if not (0 <= iid < len(self._snap.item_counts)):
                return 0
            return int(self._snap.item_counts[iid])

    def is_stale(self) -> bool:
        """快照年龄超过 max_staleness（用注入时钟，便于测试与模拟）。"""
        with self._lock:
            return (self._clock() - self._snap.created_at) > self._max_staleness

    def refresh(self, user_feat: Optional[np.ndarray] = None,
                item_feat: Optional[np.ndarray] = None,
                item_counts: Optional[np.ndarray] = None) -> int:
        """发布新快照（版本+1），清空读缓存。返回新版本号。"""
        with self._lock:
            self._snap = Snapshot(
                version=self._snap.version + 1,
                created_at=self._clock(),
                user_feat=np.ascontiguousarray(
                    user_feat if user_feat is not None else self._snap.user_feat),
                item_feat=np.ascontiguousarray(
                    item_feat if item_feat is not None else self._snap.item_feat),
                item_counts=np.asarray(
                    item_counts if item_counts is not None else self._snap.item_counts),
            )
            self._u_cache.clear()
            self._i_cache.clear()
            return self._snap.version

    @property
    def version(self) -> int:
        with self._lock:
            return self._snap.version

    def cache_report(self) -> dict:
        with self._lock:
            u = self.stats
            return {
                "version": self._snap.version,
                "age_s": round(self._snap.age_s, 1),
                "stale": self.is_stale(),
                "user_hit_rate": round(u["user_hits"] / max(1, u["user_hits"] + u["user_misses"]), 4),
                "item_hit_rate": round(u["item_hits"] / max(1, u["item_hits"] + u["item_misses"]), 4),
                "cached": {"user": len(self._u_cache), "item": len(self._i_cache)},
            }

    # ---------------- 内部 ----------------

    def _evict(self, cache: "OrderedDict[int, np.ndarray]") -> None:
        while len(cache) > self._cache_size:
            cache.popitem(last=False)


def build_from_processed(proc_dir: Path, **kw) -> FeatureStore:
    """从 data/processed 物化初始快照（延迟加载由调用方控制）。"""
    import torch  # 局部导入：服务进程按需加载

    uf = np.load(proc_dir / "user_feat.npy", allow_pickle=True) \
        if (proc_dir / "user_feat.npy").exists() else None
    # processed 里没有单独存特征 npy：从 sequences/users.csv 重建成本高，
    # 实际部署中快照由 01_prepare_data 的产出直接物化。这里用 torch_data
    # 的现成加载（其内部已是标准化张量）。
    import sys
    root = proc_dir.parent.parent
    sys.path.insert(0, str(root / "src"))
    from recsys.data.torch_data import load_processed  # noqa: E402

    d = load_processed(proc_dir)
    return FeatureStore(
        d.user_feat.numpy(), d.item_feat.numpy(), d.item_counts, **kw)
