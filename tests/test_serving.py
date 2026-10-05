"""serving 生产化模块的单元测试（护栏/缓存/实验/指标/反馈/注册表/特征）。"""

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from recsys.serving.cache import CandidateCache
from recsys.serving.experiment import Experiments, LayerConfig
from recsys.serving.feedback import EventLog
from recsys.serving.guardrails import ExposureHistory, GuardContext, Guardrails
from recsys.serving.metrics import DriftDetector, Registry
from recsys.serving.registry import ModelRegistry


# ---------------- 护栏 ----------------

class TestGuardrails:
    def _ctx(self, seen=(), now=1000.0, recent_exposure=None, freq_window=None):
        return GuardContext(user_id=1, seen=set(seen), now=now,
                            recent_exposure=dict(recent_exposure or {}),
                            freq_window=dict(freq_window or {}))

    def test_seen_and_dedup(self):
        g = Guardrails()
        res = g.apply([5, 5, 7, 9], self._ctx(seen={7}))
        assert res.passed == [5, 9]
        rules = [b["rule"] for b in res.blocked]
        assert "dedup" in rules and "seen" in rules

    def test_blocklist(self):
        g = Guardrails(blocklist={3})
        res = g.apply([1, 3, 4], self._ctx())
        assert res.passed == [1, 4]

    def test_quality_gate_cold_exempt(self):
        q = {1: 0.2, 2: 9.0}
        g = Guardrails(min_quality=5.0, quality_of=lambda i: q.get(i, 9.0),
                       cold_items={2})
        # 1 低质被拦；2 低质但冷启豁免
        res = g.apply([1, 2], self._ctx())
        assert 2 in res.passed and 1 not in res.passed

    def test_freq_cap_sliding_window(self):
        g = Guardrails(freq_cap=2, freq_window_s=100.0)
        ctx = self._ctx(now=1000.0)
        g.apply([8], ctx)               # 第 1 次入窗
        g.apply([8], ctx)               # 第 2 次入窗
        res = g.apply([8], ctx)         # 第 3 次：超 cap
        assert 8 not in res.passed
        ctx2 = self._ctx(now=1200.0)    # 窗口滑走
        assert 8 in g.apply([8], ctx2).passed

    def test_exposure_dampen_and_saturate(self):
        g = Guardrails(max_exposure_rounds=3, exposure_dampen=0.5)
        ctx = self._ctx(recent_exposure={5: 2})
        scores = {5: 1.0, 6: 0.6}
        res = g.apply([5, 6], ctx, scores=scores)
        assert 5 in res.passed
        assert math.isclose(scores[5], 1.0 * 0.25)   # 降权 0.5^2
        ctx3 = self._ctx(recent_exposure={5: 5})
        assert 5 not in g.apply([5], ctx3).passed     # 饱和拦截

    def test_exposure_history_window(self):
        h = ExposureHistory(window_requests=2)
        h.record(1, [11, 12])
        h.record(1, [11])
        h.record(1, [11])   # 第三轮，滑掉最老
        c = h.exposure_counts(1)
        assert c.get(11, 0) == 2 and c.get(12, 0) == 0   # 12 已滑出窗口（被清理）


# ---------------- 缓存 ----------------

class TestCache:
    def test_hit_miss_ttl(self):
        t = [0.0]
        c = CandidateCache(capacity=4, ttl_s=10.0, clock=lambda: t[0])
        assert c.get(1, 0) is None
        c.put(1, [1, 2, 3], 0, recall_cost_ms=25.0)
        assert c.get(1, 0) == [1, 2, 3]
        t[0] = 11.0                       # 过期
        assert c.get(1, 0) is None
        r = c.report()
        assert r["hits"] == 1 and r["expired"] == 1
        assert r["saved_ms_total"] == 25.0  # 命中结算省下的召回耗时

    def test_strategy_version_invalidates(self):
        c = CandidateCache()
        c.put(1, "v0", 0, recall_cost_ms=1)
        assert c.get(1, strategy_version=1) is None
        assert c.get(1, strategy_version=0) == "v0"

    def test_lru_eviction(self):
        c = CandidateCache(capacity=2)
        c.put(1, "a", 0); c.put(2, "b", 0); c.get(1, 0)  # 1 变热
        c.put(3, "c", 0)                                    # 驱逐 2
        assert c.get(2, 0) is None and c.get(1, 0) == "a"

    def test_invalidate_user(self):
        c = CandidateCache()
        c.put(7, "x", 0)
        c.invalidate_user(7)
        assert c.get(7, 0) is None


# ---------------- 实验 ----------------

class TestExperiments:
    def test_config_validation(self):
        with pytest.raises(ValueError):
            LayerConfig(salt="s", buckets=100, arms={"a": 40, "b": 50})  # ≠100

    def test_assignment_stable(self):
        exp = Experiments({"L": {"salt": "s1", "buckets": 100,
                                 "arms": {"control": 50, "treat": 50}}})
        assert exp.assignment(42, "L") == exp.assignment(42, "L")

    def test_split_roughly_half(self):
        exp = Experiments({"L": {"salt": "s1", "buckets": 100,
                                 "arms": {"control": 50, "treat": 50}}})
        rep = exp.layer_report("L", list(range(6000)))
        assert 0.44 < rep["control"] < 0.56

    def test_layers_independent(self):
        exp = Experiments({
            "L1": {"salt": "s1", "buckets": 10, "arms": {"a": 5, "b": 5}},
            "L2": {"salt": "s2", "buckets": 10, "arms": {"x": 5, "y": 5}},
        })
        # L1 与 L2 的分配应统计独立（联合分布近似乘积）
        n = 2000
        joint = {("a", "x"): 0, ("a", "y"): 0, ("b", "x"): 0, ("b", "y"): 0}
        for u in range(n):
            joint[(exp.assignment(u, "L1"), exp.assignment(u, "L2"))] += 1
        for k, v in joint.items():
            assert 0.16 < v / n < 0.34      # ~0.25 ± 抖动


# ---------------- 指标 ----------------

class TestMetrics:
    def test_percentiles(self):
        r = Registry()
        for v in range(1, 101):
            r.observe_ms("rank", float(v))
        assert r.percentile("rank", 0.5) <= 55
        assert r.percentile("rank", 0.95) >= 80
        assert r.percentile("rank", 0.99) >= 80

    def test_snapshot_and_prometheus(self):
        r = Registry()
        r.incr("requests")
        r.observe_ms("recall", 12.5)
        snap = r.snapshot()
        assert snap["counters"]["requests"] == 1
        text = r.expose_prometheus()
        assert "recsys_requests_total" in text and "recsys_latency_recall" in text

    def test_psi_detects_drift(self):
        d = DriftDetector(name="score", baseline_n=50, window=50)
        import random
        random.seed(0)
        for _ in range(60):
            d.push(random.uniform(0.4, 0.6))     # 基线
        for _ in range(60):
            d.push(random.uniform(0.0, 0.2))     # 突然分布漂移
        psi = d.check()
        assert psi > 0.25 and d.drift


# ---------------- 反馈闭环 ----------------

class TestFeedback:
    def test_event_log_join_and_ctr(self, tmp_path: Path):
        log = EventLog(path=tmp_path / "ev.jsonl")
        log.log("r1", 1, 10, 0, "impression", model="din", arm="control")
        log.log("r1", 1, 11, 1, "impression", model="din", arm="control")
        log.log("r1", 1, 10, 0, "click", model="din", arm="control")
        ctr = log.online_ctr()
        assert ctr["impressions"] == 2 and ctr["clicks"] == 1
        assert math.isclose(ctr["ctr"], 0.5)

    def test_dedup_and_persistence(self, tmp_path: Path):
        p = tmp_path / "ev.jsonl"
        log = EventLog(path=p)
        log.log("r1", 1, 10, 0, "impression")
        log.log("r1", 1, 10, 0, "impression")   # 幂等丢弃
        assert len(log.events()) == 1
        log.flush()                               # 批量落盘（攒批缓冲）
        log2 = EventLog(path=p)                   # 重建加载
        assert len(log2.events()) == 1

    def test_position_bias_and_snips(self):
        log = EventLog()
        for pos in range(3):
            log.log("r1", 1, 100 + pos, pos, "impression")
            if pos < 2:
                log.log("r1", 1, 100 + pos, pos, "click")  # 坑位 0/1 点击
        pb = log.position_bias()
        assert pb[0]["ctr"] == 1.0 and pb[2]["ctr"] == 0.0
        s = log.snips()
        assert s is not None and 0 < s <= 1

    def test_arm_report(self):
        log = EventLog()
        for i in range(10):
            arm = "control" if i % 2 == 0 else "treat"
            log.log(f"r{i}", i, 5, 0, "impression", arm=arm)
            if arm == "treat":
                log.log(f"r{i}", i, 5, 0, "click", arm=arm)
        rep = log.arm_report()
        assert rep["treat"]["ctr"] == 1.0 and rep["control"]["ctr"] == 0.0


# ---------------- 模型注册表 ----------------

class TestRegistry:
    def test_register_and_flow(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "reg.json")
        reg.register("din", weights="checkpoints/rank_din.pt",
                     metrics={"auc": 0.7295})
        reg.to_staging("din", shadow=True)
        reg.activate("din", note="首发上线")
        assert reg.production() == "din"
        # 持久化重建
        reg2 = ModelRegistry(path=tmp_path / "reg.json")
        assert reg2.production() == "din"

    def test_switch_and_rollback(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "reg.json")
        reg.register("din", weights="w1")
        reg.register("esmm", weights="w2")
        reg.to_staging("din")
        reg.activate("din")
        reg.to_staging("esmm")
        reg.activate("esmm", note="上线 ESMM")
        assert reg.production() == "esmm"
        back = reg.rollback()
        assert back == "din" and reg.production() == "din"

    def test_stage_guard(self, tmp_path: Path):
        reg = ModelRegistry(path=tmp_path / "reg.json")
        reg.register("m1", weights="w")
        with pytest.raises(Exception):
            reg.activate("m1")          # 未 staging 直上生产 → 拒绝
        with pytest.raises(Exception):
            reg.activate("ghost")       # 未注册


# ---------------- 特征存储 ----------------

class TestFeatureStore:
    def test_get_and_stale(self):
        import numpy as np
        from recsys.serving.feature_store import FeatureStore
        t = [0.0]
        fs = FeatureStore(np.zeros((3, 2)), np.ones((4, 2)),
                          np.array([1, 5, 2, 0]), max_staleness=10.0,
                          clock=lambda: t[0])
        v, ver, stale = fs.get_user(1)
        assert v is not None and ver == 1 and not stale
        t[0] = 11.0
        assert fs.is_stale()

    def test_lru_cache_evicts(self):
        import numpy as np
        from recsys.serving.feature_store import FeatureStore
        fs = FeatureStore(np.zeros((3, 2)), np.zeros((100, 2)),
                          np.zeros(100), cache_size=2)
        for i in (1, 2, 3):           # 逐出 1
            fs.get_item(i)
        fs.get_item(4)
        rep = fs.cache_report()
        assert rep["cached"]["item"] == 2

    def test_bad_index_returns_none(self):
        import numpy as np
        from recsys.serving.feature_store import FeatureStore
        fs = FeatureStore(np.zeros((3, 2)), np.zeros((4, 2)), np.zeros(4))
        v, _, _ = fs.get_user(99)
        assert v is None
