#!/usr/bin/env python
"""Step 17：服务并发压测（真实 HTTP 负载，非进程内计时）。

对运行中的 webapp 发起并发请求（标准库 ThreadPool + 真实 HTTP 往返），
度量：总吞吐（QPS）、p50/p95/p99 端到端延迟、错误率、缓存命中时的
延迟差异。这是对「服务进程 + 网络 + 序列化」全链路的真实测量。

用法：
    # 先确保服务在跑：python webapp/server.py
    python scripts/17_load_test.py [--n 300] [--concurrency 8] [--base http://127.0.0.1:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from recsys.common import ROOT as PROJ_ROOT, get_logger, save_json  # noqa: E402

log = get_logger("load_test")


def hit(api: str, params: dict) -> tuple[float, int]:
    """一次真实 HTTP 请求，返回 (耗时 ms, 状态码)。"""
    url = f"{api}?{urlencode(params)}"
    t0 = time.perf_counter()
    try:
        with urlopen(url, timeout=30) as r:
            r.read()
            code = r.status
    except Exception as e:  # noqa: BLE001
        code = getattr(e, "code", 0)
    return (time.perf_counter() - t0) * 1000, code


def pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if len(arr) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="总请求数")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--base", type=str, default="http://127.0.0.1:8000")
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    # 健康检查
    _, code = hit(f"{args.base}/api/health", {})
    if code != 200:
        log.error("服务不可用（/api/health=%s）。先启动: python webapp/server.py", code)
        sys.exit(1)

    # 热身一轮（首次请求含缓存 miss 的全量召回 + torch 线程池扩张）
    for u in (7, 42):
        hit(f"{args.base}/api/recommend", {"user": u, "model": "auto", "topn": args.topn})

    # 用户混合：一半真实分布（幂律热门用户），一半长尾随机 → 覆盖冷热路径
    hot_users = (rng.power(2.0, args.n) * 6039 + 1).astype(int)
    tail_users = rng.integers(0, 6040, args.n)
    users = np.where(np.arange(args.n) % 2 == 0, hot_users, tail_users)

    lat, codes, t_start = [], [], time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(hit, f"{args.base}/api/recommend",
                            {"user": int(u), "model": "auto", "topn": args.topn})
                for u in users]
        for f in futs:
            ms, code = f.result()
            lat.append(ms)
            codes.append(code)
    wall = time.perf_counter() - t_start

    lat = np.asarray(lat)
    ok = [c == 200 for c in codes]
    # 命中缓存的请求（偶发同用户重复）显著更快——分桶观察
    out = {
        "base": args.base, "requests": args.n, "concurrency": args.concurrency,
        "wall_s": round(wall, 2),
        "qps": round(args.n / wall, 1),
        "ok": int(sum(ok)), "errors": int(args.n - sum(ok)),
        "latency_ms": {
            "mean": round(float(lat[ok].mean()) if sum(ok) else 0, 1),
            "p50": round(pct(lat[ok], 50), 1),
            "p95": round(pct(lat[ok], 95), 1),
            "p99": round(pct(lat[ok], 99), 1),
            "max": round(float(lat[ok].max()) if sum(ok) else 0, 1),
        },
    }
    save_json(out, PROJ_ROOT / "results" / "load_test.json")
    log.info("QPS %.1f（%d 请求 / %.1fs，并发 %d）p50/p95/p99 = %.1f/%.1f/%.1f ms，错误 %d",
             out["qps"], args.n, wall, args.concurrency,
             out["latency_ms"]["p50"], out["latency_ms"]["p95"],
             out["latency_ms"]["p99"], out["errors"])
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
