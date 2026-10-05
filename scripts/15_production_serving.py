#!/usr/bin/env python
"""Step 15：生产化链路演练（serving 全栈冒烟测试）。

不启动 HTTP 服务，直接驱动 webapp/server.py 的 RecSysState（复用其
完整请求管线：实验分桶 → 候选缓存 → 护栏 → 精排 → 重排 → 曝光事件流
→ 指标/漂移），模拟一次「上线之夜」：

1. 流量回放：600 次请求（70% 实验分桶 auto / 30% 手动模型），
   同用户重复请求触发候选缓存命中；
2. 用户反馈模拟：按位置衰减的点击概率 + 少量点赞/点踩
   （曝光-点击进 EventLog，可回放）；
3. 模型切换演练：esmm staging → 上线 → 一键回滚；
4. 产出：results/production_serving.json（指标/缓存/漂移/分桶 CTR/
   位置偏差/SNIPS/注册表历史）。

用法：
    python scripts/15_production_serving.py [--requests 600]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from recsys.common import get_logger  # noqa: E402

log = get_logger("prod_drill")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    t0 = time.perf_counter()
    from webapp.server import RecSysState  # 复用完整请求管线
    state = RecSysState()
    log.info("状态加载完成 %.0fs，开始流量回放", time.perf_counter() - t0)

    n = max(50, args.requests)
    users = [rng.randrange(state.num_users) for _ in range(n)]
    # 前 1/3 的请求用已见过的用户 → 驱动缓存命中
    users[: n // 3] = [rng.choice(users[n // 3:]) for _ in range(n // 3)]

    n_click = n_like = n_dis = 0
    cache_hits = 0
    for i, uid in enumerate(users):
        mode = "auto" if rng.random() < 0.7 else rng.choice(["din", "esmm", "deepfm"])
        res = state.recommend(uid, model=mode, topn=10,
                              lam=round(rng.uniform(0.5, 0.9), 1),
                              rerank=True, cold_quota=rng.choice([0, 0, 3, 5]))
        if res.get("error"):
            continue
        cache_hits += 1 if res["cache_hit"] else 0

        # 位置衰减的点击模拟：P(click) ∝ 0.42 * 0.72^pos（演示反馈闭环用，
        # 真实线上点击来自用户）
        for pos, c in enumerate(res["items"]):
            p = 0.42 * (0.72 ** pos)
            if rng.random() < p:
                state.log_feedback(res["request_id"], uid, c["item"], pos, "click")
                n_click += 1
                r = rng.random()
                if r < 0.35:
                    state.log_feedback(res["request_id"], uid, c["item"], pos, "like")
                    n_like += 1
                elif r > 0.9:
                    state.log_feedback(res["request_id"], uid, c["item"], pos, "dislike")
                    n_dis += 1
                break
        if (i + 1) % 200 == 0:
            log.info("进度 %d/%d，缓存命中 %d", i + 1, n, cache_hits)

    # —— 模型切换演练：esmm 影子验证 → 上线 → 回滚 ——
    state.registry.to_staging("esmm", shadow=True)
    state.registry.activate("esmm", note="演练：ESMM 上线")
    prod_after = state.registry.production()
    rolled = state.registry.rollback()
    log.info("切换演练: esmm 上线(%s) → 回滚到(%s)", prod_after, rolled)

    ops = state.ops_report()
    fb = state.feedback_report()
    exp = state.experiment_info(7)

    out = {
        "requests": n, "cache_hits": cache_hits,
        "simulated_feedback": {"clicks": n_click, "likes": n_like, "dislikes": n_dis},
        "ops": ops,
        "feedback": {
            "online_ctr": fb["online_ctr"],
            "position_bias": fb["position_bias"],
            "snips": fb["snips"],
            "arms": fb["arms"],
        },
        "experiment": exp,
        "registry_drill": {"activated": prod_after, "rolled_back_to": rolled},
        "prometheus_head": "\n".join(state.metrics.expose_prometheus().splitlines()[:8]),
    }
    (ROOT / "results" / "production_serving.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    m = ops["metrics"]
    print("\n===== 生产链路演练报告 =====")
    print(f"请求 {n} 次（缓存命中 {cache_hits}，命中率 "
          f"{ops['cache']['hit_rate']:.1%}，节省召回 {ops['cache']['saved_ms_total']:.0f}ms）")
    print(f"延迟 p50/p95/p99: {m['latency']['total']['p50']:.1f} / "
          f"{m['latency']['total']['p95']:.1f} / {m['latency']['total']['p99']:.1f} ms")
    print(f"漂移检测 PSI: {ops['drift']['psi']}（drift={ops['drift']['drift']}）")
    ctr = fb["online_ctr"]
    print(f"在线 CTR: {ctr['ctr']:.1%}（{ctr['clicks']}/{ctr['impressions']}）"
          if ctr.get("impressions") else "在线 CTR: n/a")
    for a, v in fb["arms"].items():
        if v.get("impressions"):
            print(f"  arm {a:14s} ctr={v['ctr']:.1%}  imps={v['impressions']}"
                  f"  p50={v.get('latency_p50_ms')}ms")
    pb = fb["position_bias"][:5]
    print("位置偏差（前 5 坑）:", [(p["pos"], round(p["ctr"], 3)) for p in pb])
    print(f"SNIPS 去偏 CTR: {fb['snips']}")
    print(f"注册表演练: {prod_after} 上线 → 回滚 {rolled}")
    print("结果已写入 results/production_serving.json")


if __name__ == "__main__":
    main()
