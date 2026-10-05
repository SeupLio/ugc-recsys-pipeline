"""可观测性（Metrics）—— QPS / 延迟分位数 / 错误率 / PSI 漂移 / Prometheus 格式。

线上系统的「第二双眼」：没有指标就没有运营。本模块提供：

- Registry：线程安全的计数器 + 延迟直方（bucket 采样，p50/p95/p99）
- observe(stage, ms) / incr(name) / observe_value(name, v)
- DriftDetector：PSI（Population Stability Index）监控分数分布漂移，
  基线 vs 近窗，>0.25 报警（业界惯例阈值）
- expose_prometheus()：文本格式暴露，可被 Prometheus 抓取（生产替换
  /api/metrics 的数据源即可，协议不变）

直方实现：对数分桶（1ms→10s），内存 O(bucket 数)，分位数用桶内线性插值。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

# 对数桶边界：1ms 到 10s
_EDGES = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500,
                   1000, 2000, 5000, 10000], dtype=np.float64)

_PSI_BINS = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
                      0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)


class Registry:
    def __init__(self, clock=time.time) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._counters: Dict[str, int] = defaultdict(int)
        self._histos: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(len(_EDGES) + 1, dtype=np.int64))
        self._values: Dict[str, List[float]] = defaultdict(list)
        self._started = clock()

    # ---------------- 记录 ----------------

    def incr(self, name: str, delta: int = 1) -> None:
        with self._lock:
            self._counters[name] += delta

    def observe_ms(self, stage: str, ms: float) -> None:
        with self._lock:
            idx = int(np.searchsorted(_EDGES, ms, side="right"))
            self._histos[stage][idx] += 1

    def observe_value(self, name: str, v: float, keep: int = 4096) -> None:
        with self._lock:
            buf = self._values[name]
            buf.append(float(v))
            if len(buf) > keep:
                del buf[: len(buf) - keep]

    # ---------------- 读取 ----------------

    def percentile(self, stage: str, q: float) -> float:
        """桶内线性插值分位数。空数据返回 0。"""
        with self._lock:
            h = self._histos.get(stage)
            if h is None or h.sum() == 0:
                return 0.0
            h = np.asarray(h, dtype=np.float64)
            target = q * h.sum()
            cum = np.cumsum(h)
            b = int(np.searchsorted(cum, target))
            b = min(b, len(_EDGES))
            lo = _EDGES[b - 1] if b > 0 else 0.0
            hi = _EDGES[b] if b < len(_EDGES) else _EDGES[-1] * 2
            c_prev = cum[b - 1] if b > 0 else 0.0
            in_bucket = h[b]
            frac = (target - c_prev) / in_bucket if in_bucket > 0 else 0.0
            return float(lo + frac * (hi - lo))

    def snapshot(self) -> dict:
        with self._lock:
            out = {
                "uptime_s": round(self._clock() - self._started, 1),
                "counters": dict(self._counters),
                "latency": {},
                "qps": {},
            }
            total = self._counters.get("requests", 0)
            up = max(1e-9, out["uptime_s"])
            for stage, h in self._histos.items():
                n = int(h.sum())
                out["latency"][stage] = {
                    "count": n,
                    "p50": round(self.percentile(stage, 0.50), 2),
                    "p95": round(self.percentile(stage, 0.95), 2),
                    "p99": round(self.percentile(stage, 0.99), 2),
                }
            out["qps"]["requests"] = round(total / up, 3)
            return out

    def expose_prometheus(self) -> str:
        """Prometheus 文本格式（type/样本行）。"""
        with self._lock:
            lines: List[str] = []
            for name, v in sorted(self._counters.items()):
                lines.append(f"recsys_{name}_total {v}")
            for stage, h in self._histos.items():
                s = stage.replace("-", "_")
                lines.append(f"recsys_latency_{s}_count {int(h.sum())}")
                for q in (0.5, 0.95, 0.99):
                    lines.append(
                        f'recsys_latency_{s}{{quantile="{q}"}} '
                        f"{self.percentile(stage, q):.2f}")
            return "\n".join(lines) + "\n"


class DriftDetector:
    """PSI 漂移检测：监控精排分数分布的稳定性。

    基线 = 前 `baseline_n` 条观测的分桶占比；之后每 check 一次，
    用近 `window` 条与基线算 PSI。>0.25 判定漂移（需要人工介入）。
    """

    def __init__(self, name: str = "rank_score", baseline_n: int = 200,
                 window: int = 200) -> None:
        self._name = name
        self._baseline_n = int(baseline_n)
        self._window = int(window)
        self._baseline: np.ndarray | None = None
        self._recent: List[float] = []
        self.last_psi = 0.0
        self.drift = False

    def push(self, v: float) -> None:
        if self._baseline is None:
            self._recent.append(v)
            if len(self._recent) >= self._baseline_n:
                self._baseline = self._bin(self._recent)
                self._recent = []
        else:
            self._recent.append(v)
            if len(self._recent) > self._window:
                self._recent = self._recent[-self._window:]

    def check(self) -> float:
        if self._baseline is None or len(self._recent) < self._window:
            return self.last_psi
        cur = self._bin(self._recent)
        base = np.maximum(self._baseline, 1e-6)
        cur = np.maximum(cur, 1e-6)
        self.last_psi = float(((cur - base) * np.log(cur / base)).sum())
        self.drift = self.last_psi > 0.25
        return self.last_psi

    def status(self) -> dict:
        return {"metric": self._name, "psi": round(self.last_psi, 4),
                "drift": self.drift,
                "state": "baseline" if self._baseline is None else "monitoring",
                "threshold": 0.25}

    @staticmethod
    def _bin(vals: List[float]) -> np.ndarray:
        arr = np.clip(np.asarray(vals, dtype=np.float64), 0.0, 1.0)
        counts, _ = np.histogram(arr, bins=np.concatenate([[0.0], _PSI_BINS, [1.0]]))
        return counts / max(1, len(arr))
