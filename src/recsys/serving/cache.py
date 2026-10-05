"""多级缓存 —— 候选级 LRU+TTL，附带命中率与延迟节省统计。

推荐请求有天然的缓存机会：同一用户短时间内的召回结果高度稳定。
但缓存推荐列表有「信息茧房」风险，所以本实现缓存的是「召回候选集」
而非「最终列表」——精排/重排每轮重算，个性化与多样性不受缓存影响。

CandidateCache:
- key = (user_id, strategy_version)   策略变更自动失效
- value = 召回候选 + 通道归因
- TTL 过期 + LRU 容量驱逐 + 线程安全
- report() 输出命中率与累计节省的召回毫秒数（缓存 ROI）
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class CandidateCache:
    """(user, strategy) → 召回候选 的 TTL+LRU 缓存。

    put 时登记该条目的召回成本 recall_cost_ms；命中时把成本累进
    saved_ms——「这一缓存替我省了多少毫秒的召回计算」。
    """

    def __init__(self, capacity: int = 1024, ttl_s: float = 120.0,
                 clock=time.time) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._store: "OrderedDict[Tuple[int, int], Tuple[float, Any]]" = OrderedDict()
        self._costs: Dict[Tuple[int, int], float] = {}
        self._cap = int(capacity)
        self._ttl = float(ttl_s)
        self.stats = {"hits": 0, "misses": 0, "expired": 0,
                      "evictions": 0, "saved_ms": 0.0}

    def get(self, user_id: int, strategy_version: int = 0) -> Optional[Any]:
        key = (int(user_id), int(strategy_version))
        with self._lock:
            ent = self._store.get(key)
            if ent is None:
                self.stats["misses"] += 1
                return None
            ts, val = ent
            if self._clock() - ts > self._ttl:
                self._store.pop(key, None)
                self._costs.pop(key, None)
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                return None
            self._store.move_to_end(key)
            self.stats["hits"] += 1
            self.stats["saved_ms"] += self._costs.get(key, 0.0)
            return val

    def put(self, user_id: int, val: Any, strategy_version: int = 0,
            recall_cost_ms: float = 0.0) -> None:
        key = (int(user_id), int(strategy_version))
        with self._lock:
            self._store[key] = (self._clock(), val)
            self._costs[key] = float(recall_cost_ms)
            self._store.move_to_end(key)
            while len(self._store) > self._cap:
                ev = self._store.popitem(last=False)[0]
                self._costs.pop(ev, None)
                self.stats["evictions"] += 1

    def invalidate_user(self, user_id: int) -> None:
        """用户产生新行为后作废其候选缓存（保持新鲜度）。"""
        with self._lock:
            for key in [k for k in self._store if k[0] == int(user_id)]:
                self._store.pop(key, None)
                self._costs.pop(key, None)

    def report(self) -> Dict[str, float]:
        with self._lock:
            total = self.stats["hits"] + self.stats["misses"]
            return {
                "hit_rate": round(self.stats["hits"] / max(1, total), 4),
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "expired": self.stats["expired"],
                "evictions": self.stats["evictions"],
                "entries": len(self._store),
                "saved_ms_total": round(self.stats["saved_ms"], 1),
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._costs.clear()
            self.stats = {"hits": 0, "misses": 0, "expired": 0,
                          "evictions": 0, "saved_ms": 0.0}
