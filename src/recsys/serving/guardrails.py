"""安全护栏（Guardrails）—— 推荐结果出闸前的最后一道防线。

生产事故里召回/排序本身很少翻车，翻车的多是「没过滤」：
重复曝光惹恼用户、下架内容还在推、低质内容冲高 CTR 但毁生态。
本模块把常见护栏做成可组合的过滤器链：

    apply_guardrails(candidates, ctx) -> FilterResult

已实现：
1. seen_filter      ：用户已看过 / 已交互（隐式反馈去重）
2. exposure_dampen  ：近期曝光未点击的降权（探索新鲜度，防「老面孔霸屏」）
3. frequency_cap    ：单内容对单用户的频控（滑动窗口计数）
4. blocklist        ：黑名单（下架/审核中/版权撤回）
5. quality_gate     ：质量分门槛（低于 min_quality 拦截，冷启豁免可配）
6. dedup            ：重复 item 只保留一个（防御上游 bug）

FilterResult 保留完整审计轨迹（每条被拦记录原因），
线上排查「为什么这条没推出来」直接看 trail。
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


@dataclass
class GuardContext:
    """一次请求的护栏上下文。"""

    user_id: int
    seen: Set[int] = field(default_factory=set)              # 已交互物品
    recent_exposure: Dict[int, int] = field(default_factory=dict)  # iid -> 近 K 次请求曝光次数
    freq_window: Dict[int, deque] = field(default_factory=dict)    # iid -> 时间戳队列
    now: float = 0.0
    request_id: str = ""


@dataclass
class FilterResult:
    passed: List[int] = field(default_factory=list)
    blocked: List[dict] = field(default_factory=list)   # [{"item":..,"rule":..}]

    @property
    def blocked_count(self) -> int:
        return len(self.blocked)

    def trail(self, top: int = 10) -> List[dict]:
        """审计轨迹（前 top 条拦截记录）。"""
        return self.blocked[:top]


class Guardrails:
    """可配置护栏链。所有阈值集中在构造参数，等价于线上「策略配置」。"""

    def __init__(self, *,
                 blocklist: Optional[Set[int]] = None,
                 min_quality: float = 0.0,
                 quality_of: Optional[Callable[[int], float]] = None,
                 cold_items: Optional[Set[int]] = None,
                 freq_cap: int = 2,
                 freq_window_s: float = 3600.0,
                 exposure_dampen: float = 0.5,
                 max_exposure_rounds: int = 3,
                 enable: Iterable[str] = ("seen", "freq", "blocklist",
                                          "quality", "dedup", "exposure")) -> None:
        self.blocklist = set(blocklist or set())
        self.min_quality = float(min_quality)
        self.quality_of = quality_of or (lambda _i: 1.0)
        self.cold_items = set(cold_items or set())
        self.freq_cap = int(freq_cap)
        self.freq_window_s = float(freq_window_s)
        self.exposure_dampen = float(exposure_dampen)
        self.max_exposure_rounds = int(max_exposure_rounds)
        self.enable = set(enable)

    # ---------------- 主入口 ----------------

    def apply(self, candidates: Sequence[int], ctx: GuardContext,
              scores: Optional[Dict[int, float]] = None,
              enable: Optional[Iterable[str]] = None) -> FilterResult:
        """顺序过闸。candidates 按相关度降序，返回保持原顺序的存活集。

        scores 可选：exposure_dampen 会改写传入 dict 里的分值（就地降权），
        让「降权」而不是「拦截」类规则也能影响最终排序。
        enable 可选：本次生效的规则子集（护栏降级时跳过频控/曝光规则）。
        """
        rules = set(enable) if enable is not None else self.enable
        scores = scores if scores is not None else {}
        res = FilterResult()
        seen_once: Set[int] = set()

        for iid in candidates:
            iid = int(iid)

            if "dedup" in rules and iid in seen_once:
                res.blocked.append({"item": iid, "rule": "dedup"})
                continue
            seen_once.add(iid)

            if "seen" in rules and iid in ctx.seen:
                res.blocked.append({"item": iid, "rule": "seen"})
                continue

            if "blocklist" in rules and iid in self.blocklist:
                res.blocked.append({"item": iid, "rule": "blocklist"})
                continue

            if "freq" in rules:
                q = ctx.freq_window.setdefault(iid, deque())
                while q and ctx.now - q[0] > self.freq_window_s:
                    q.popleft()
                if len(q) >= self.freq_cap:
                    res.blocked.append({"item": iid, "rule": f"freq_cap({self.freq_cap})"})
                    continue
                q.append(ctx.now)

            if "quality" in rules:
                q_score = self.quality_of(iid)
                is_cold = iid in self.cold_items
                if q_score < self.min_quality and not is_cold:
                    res.blocked.append({
                        "item": iid, "rule": f"quality<{self.min_quality:.1f}({q_score:.1f})"})
                    continue

            # 曝光降权（不拦截，改分数，允许靠强相关分冲回来）
            if "exposure" in rules and iid in ctx.recent_exposure:
                rounds = ctx.recent_exposure[iid]
                if rounds >= self.max_exposure_rounds:
                    res.blocked.append({"item": iid, "rule": "exposure_saturate"})
                    continue
                if iid in scores:
                    scores[iid] *= self.exposure_dampen ** rounds

            res.passed.append(iid)
        return res

    def record_impression(self, ctx: GuardContext, items: Sequence[int]) -> None:
        """请求结束后回写曝光计数（供下一轮 exposure_dampen 使用）。"""
        for iid in items:
            ctx.recent_exposure[int(iid)] = ctx.recent_exposure.get(int(iid), 0) + 1


class ExposureHistory:
    """跨请求的曝光记忆（user_id -> iid -> 近 N 轮曝光数）。

    webapp 进程内常驻，模拟线上 Redis 的 short-term 曝光窗口。
    窗口语义：按「请求轮次」滑动——每个 item 记录出现过的轮次号，
    统计时只数最近 window_requests 轮。上一轮出现、最近 N 轮未再出现
    的 item 计数自然衰减到 0（这正是「防老面孔霸屏」想要的行为）。
    """

    def __init__(self, max_users: int = 10000, window_requests: int = 5) -> None:
        self._lock = threading.RLock()
        self._rounds: Dict[int, int] = defaultdict(int)       # user -> 当前轮次
        self._data: Dict[int, Dict[int, deque]] = defaultdict(dict)
        self._window = int(window_requests)
        self._max_users = max_users

    def exposure_counts(self, user_id: int) -> Dict[int, int]:
        """返回该用户每个 item 的近 N 轮曝光次数。"""
        with self._lock:
            cur = self._rounds[user_id]
            lo = cur - self._window + 1
            return {iid: sum(1 for r in q if r >= lo)
                    for iid, q in self._data[user_id].items()}

    def record(self, user_id: int, items: Iterable[int]) -> None:
        """一次请求的展示列表入账（轮次 +1）。"""
        with self._lock:
            self._rounds[user_id] += 1
            r = self._rounds[user_id]
            per_item = self._data[user_id]
            for iid in items:
                q = per_item.setdefault(int(iid), deque(maxlen=self._window))
                q.append(r)
            # 顺手清理窗口外仍为 0 计数的条目，控制内存
            lo = r - self._window
            for iid in [k for k, q in per_item.items() if q and q[-1] <= lo]:
                per_item.pop(iid, None)
            # 简单容量控制：超限驱逐最老用户
            if len(self._data) > self._max_users:
                oldest = next(iter(self._data))
                self._data.pop(oldest)
                self._rounds.pop(oldest, None)
