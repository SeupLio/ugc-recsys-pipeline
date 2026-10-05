"""反馈闭环（Feedback Loop）—— 曝光-点击事件流与在线指标。

推荐系统与普通 ML 最大的差异：模型改变用户行为，行为再变成新数据。
没有反馈闭环的推荐是「一次性推荐」，不是产品。本模块实现：

- EventLog：JSONL 追加式事件流（request_id 串联曝光与点击），
  幂等去重（同 request+item+type 只记一次）、线程安全、自动限容
- 回放分析：
  * online_ctr()  ：在线 CTR（曝光/点击 join）
  * position_bias()：按坑位的 CTR 曲线（显式度量位置偏差）
  * snips()       ：SNIPS 逆倾向加权（对位置做去偏的整体指标）
  * arm_report()  ：按实验分桶聚合（在线 A/B 的读数）

数据格式（一行一个 JSON）：
  {"ts": ..., "request_id": "...", "user": 7, "item": 253, "pos": 0,
   "event": "impression" | "click" | "like" | "dislike",
   "model": "din", "arm": "control", "latency_ms": 29.5}
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class EventLog:
    """追加式事件流（进程内写，JSONL 落盘，读时全量回放）。

    落盘为批量写：事件先进 pending 缓冲，攒满 flush_every 条或显式
    flush() 时一次追加（网络盘上逐条 open/append 是主要延迟来源）。
    """

    EVENTS = ("impression", "click", "like", "dislike")

    def __init__(self, path: Optional[Path] = None, max_events: int = 200_000,
                 flush_every: int = 64) -> None:
        self._lock = threading.RLock()
        self._path = Path(path) if path else None
        self._max = int(max_events)
        self._flush_every = int(flush_every)
        self._pending: List[str] = []          # 待落盘的 JSONL 行
        self._buf: List[dict] = []
        self._seen_keys: set = set()
        self._n_dropped = 0
        if self._path and self._path.exists():
            self._load_existing()

    def flush(self) -> None:
        """把 pending 缓冲一次性落盘（进程退出前 / 定时器调用）。"""
        with self._lock:
            if self._pending and self._path:
                with self._path.open("a", encoding="utf-8") as f:
                    f.writelines(self._pending)
            self._pending.clear()

    def __del__(self) -> None:  # 尽力而为的兜底落盘
        try:
            self.flush()
        except Exception:
            pass

    # ---------------- 写 ----------------

    def log(self, request_id: str, user: int, item: int, pos: int,
            event: str, model: str = "", arm: str = "",
            latency_ms: float = 0.0, ts: Optional[float] = None) -> bool:
        if event not in self.EVENTS:
            raise ValueError(f"未知事件类型: {event}")
        key = (request_id, int(item), event)
        with self._lock:
            if key in self._seen_keys:
                return False            # 幂等：重复上报只记一次
            self._seen_keys.add(key)
            rec = {"ts": ts if ts is not None else time.time(),
                   "request_id": request_id, "user": int(user), "item": int(item),
                   "pos": int(pos), "event": event, "model": model,
                   "arm": arm, "latency_ms": round(float(latency_ms), 2)}
            self._buf.append(rec)
            if len(self._buf) > self._max:
                self._n_dropped += len(self._buf) - self._max
                del self._buf[: len(self._buf) - self._max]
            if self._path:
                self._pending.append(json.dumps(rec, ensure_ascii=False) + "\n")
                if len(self._pending) >= self._flush_every:
                    self.flush()
        return True

    # ---------------- 读 ----------------

    def events(self) -> List[dict]:
        with self._lock:
            return list(self._buf)

    def replay(self) -> Dict[str, dict]:
        """按 request_id 聚合：每条请求的曝光集合与点击集合。"""
        agg: Dict[str, dict] = {}
        for e in self.events():
            r = agg.setdefault(e["request_id"],
                               {"user": e["user"], "model": e["model"],
                                "arm": e["arm"], "latency_ms": e["latency_ms"],
                                "imps": {}, "clicks": set(), "likes": set(),
                                "dislikes": set()})
            if e["event"] == "impression":
                r["imps"][e["item"]] = e["pos"]
            elif e["event"] == "click":
                r["clicks"].add(e["item"])
            elif e["event"] == "like":
                r["likes"].add(e["item"])
            elif e["event"] == "dislike":
                r["dislikes"].add(e["item"])
        return agg

    def online_ctr(self, arm: Optional[str] = None) -> dict:
        """整体在线 CTR（可按实验 arm 过滤）。"""
        n_imp = n_clk = 0
        for r in self.replay().values():
            if arm and r["arm"] != arm:
                continue
            n_imp += len(r["imps"])
            n_clk += len(r["clicks"])
        return {"impressions": n_imp, "clicks": n_clk,
                "ctr": round(n_clk / n_imp, 4) if n_imp else None}

    def position_bias(self, top_pos: int = 10) -> List[dict]:
        """第 pos 坑位的 CTR（位置偏差的直接度量）。"""
        by_pos: Dict[int, List[int]] = defaultdict(lambda: [0, 0])  # pos -> [imp, clk]
        for r in self.replay().values():
            for item, pos in r["imps"].items():
                if pos >= top_pos:
                    continue
                by_pos[pos][0] += 1
                by_pos[pos][1] += 1 if item in r["clicks"] else 0
        return [{"pos": p, "imps": c[0], "ctr": round(c[1] / c[0], 4) if c[0] else None}
                for p, c in sorted(by_pos.items())]

    def snips(self) -> Optional[float]:
        """SNIPS：对曝光按 1/倾向(位置) 加权的 CTR，缓解位置偏差。

        倾向用经验位置 CTR 估计（自洽：数据自身估计的倾向 + 逆加权）。
        返回 None 表示点击太少无法估计。
        """
        pos_stat: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
        for r in self.replay().values():
            for item, pos in r["imps"].items():
                pos_stat[pos][0] += 1
                pos_stat[pos][1] += 1 if item in r["clicks"] else 0
        if not pos_stat:
            return None
        prop = {p: (c[0] and c[1] / c[0]) or 0.0 for p, c in pos_stat.items()}
        if not any(prop.values()):
            return None
        w_num = w_den = 0.0
        for r in self.replay().values():
            for item, pos in r["imps"].items():
                pr = prop.get(pos, 0.0)
                if pr <= 0:
                    continue
                w = 1.0 / pr
                w_den += w
                if item in r["clicks"]:
                    w_num += w
        return round(w_num / w_den, 4) if w_den > 0 else None

    def arm_report(self) -> Dict[str, dict]:
        """按 arm 聚合的在线读数（A/B 日报的雏形）。"""
        out: Dict[str, dict] = defaultdict(lambda: {"imps": 0, "clicks": 0,
                                                    "users": set(), "lat": []})
        for r in self.replay().values():
            a = out[r["arm"] or "default"]
            a["imps"] += len(r["imps"])
            a["clicks"] += len(r["clicks"])
            a["users"].add(r["user"])
            a["lat"].append(r["latency_ms"])
        res = {}
        for arm, a in out.items():
            res[arm] = {"users": len(a["users"]), "impressions": a["imps"],
                        "clicks": a["clicks"],
                        "ctr": round(a["clicks"] / a["imps"], 4) if a["imps"] else None,
                        "latency_p50_ms": round(sorted(a["lat"])[len(a["lat"]) // 2], 1)
                        if a["lat"] else None}
        return res

    def stats(self) -> dict:
        with self._lock:
            return {"events": len(self._buf), "dropped": self._n_dropped,
                    "path": str(self._path) if self._path else "(内存)"}

    # ---------------- 内部 ----------------

    def _load_existing(self) -> None:
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():  # type: ignore
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._buf.append(rec)
                self._seen_keys.add((rec["request_id"], int(rec["item"]), rec["event"]))
        except OSError:
            pass
