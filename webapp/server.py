#!/usr/bin/env python
"""本地 Web Demo 服务：召回 → 精排 → 重排 全链路交互式展示。

零第三方 Web 框架依赖（标准库 http.server），只依赖训练产物：
    data/processed/* 与 checkpoints/*（先跑 scripts/01~05）。

启动：
    python webapp/server.py            # 默认 127.0.0.1:8000
    python webapp/server.py --port 9000

设计要点：
- 所有重计算在启动时一次性完成（数据加载、ItemCF 训练、三个精排模型加载、
  向量点积预归一），单次请求只剩「查表 + 小矩阵乘」，毫秒级返回。
- 推荐 / 看了又看 / 用户画像 / 画像编码反解 全部走 REST JSON API。
- 推荐请求支持：精排模型切换(DeepFM/DIN/ESMM)、Top-N、MMR 多样性 λ、
  冷启扶持额度，并返回逐阶段延迟与召回通道归因。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STATIC = Path(__file__).resolve().parent / "static"

from recsys.common import DIR_CKPT, DIR_PROC, get_logger  # noqa: E402
from recsys.data.torch_data import build_batch, load_processed  # noqa: E402
from recsys.models.rank import DINRanker, DeepFMRanker, ESMM, RankConfig  # noqa: E402
from recsys.recall.collaborative import ItemCF, PopularityRecall  # noqa: E402
from recsys.rerank.mmr import mmr_rerank  # noqa: E402
from recsys.serving.cache import CandidateCache  # noqa: E402
from recsys.serving.experiment import Experiments  # noqa: E402
from recsys.serving.feedback import EventLog  # noqa: E402
from recsys.serving.guardrails import ExposureHistory, GuardContext, Guardrails  # noqa: E402
from recsys.serving.metrics import DriftDetector, Registry  # noqa: E402
from recsys.serving.registry import ModelRegistry  # noqa: E402

import uuid  # noqa: E402

log = get_logger("web_demo")

# ---------- 展示用的静态映射（ML-1M 官方口径） ----------
# age_idx = sorted(unique(age)) 的序号，ML-1M 共 7 桶
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
OCC_LABELS = [
    "其他/未填写", "学术界/教育家", "艺术家", "文书/行政", "大学生/研究生",
    "客户服务", "医生/医疗", "行政/管理", "农民", "家庭主妇",
    "中小学学生", "律师", "程序员", "退休", "销售/市场",
    "科学家", "个体经营者", "技师/工程师", "无业", "作家",
    "娱乐业",
]
GENDER_LABELS = {0: "F", 1: "M"}

RECALL_TOPN = 60      # 每路召回取多少
FUSE_TOPN = 80        # 融合后进入精排的候选上限
MAX_SEQ = 50          # 用户行为序列展示上限


def _age_label(age_idx: int) -> str:
    return AGE_LABELS[age_idx] if 0 <= age_idx < len(AGE_LABELS) else "?"


class RecSysState:
    """启动时把全部重状态物化，请求期只做轻量计算。"""

    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.data = load_processed(DIR_PROC)
        d = self.data
        self.num_users, self.num_items = d.num_users, d.num_items

        # —— 展示层素材 ——
        self.vocab = json.loads((DIR_PROC / "vocab.json").read_text(encoding="utf-8"))
        self.gname = {v: k for k, v in self.vocab["genre2idx"].items()}
        it = d.items_df.set_index("item_idx")
        self.titles = it["clean_title"].to_dict()
        self.genres = it["genres"].to_dict()
        self.item_stat = {  # 质量分/口碑用于卡片展示
            int(i): {"n_users": int(r["i_n_unique_users"]), "mean_rating": float(r["i_mean_rating"]),
                     "quality": float(r["i_score_decay"])}
            for i, r in it.iterrows()}
        self.cold_set = set(int(x) for x in d.cold_items) if len(d.cold_items) else set()

        # 用户展示信息（原始 users.dat 反解人口属性）
        self.user_meta = self._load_user_meta()

        # —— 召回通道（只用 train 正样本，与 09_demo_serve 完全同口径）——
        train_all = d.interactions[d.interactions["split_tag"] == "train"]
        train_pos = train_all[train_all["rating"] >= 4]
        self.hot = PopularityRecall.from_interactions(
            train_pos, d.num_items, half_life_days=30.0)
        t1 = time.perf_counter()
        self.icf = ItemCF(d.num_items, top_k_sim=100).fit(train_pos)
        log.info("ItemCF 启动预热: %.1fs", time.perf_counter() - t1)

        # —— 向量（已归一，点积即余弦） ——
        self.user_emb = np.load(DIR_PROC / "user_emb.npy").astype(np.float32)
        self.item_emb = np.load(DIR_PROC / "item_emb.npy").astype(np.float32)
        self.content_emb = np.load(DIR_PROC / "content" / "content_emb.npy").astype(np.float32)

        # —— 精排模型（三套全载，前端可切换） ——
        cfg = RankConfig(
            num_users=d.num_users, num_items=d.num_items,
            num_genres=max(d.num_genres, 1),
            num_gender=int(d.users_df["gender_idx"].max()) + 1,
            num_age=int(d.users_df["age_idx"].max()) + 1,
            num_occ=int(d.users_df["occ_idx"].max()) + 1,
        )
        self.models = {}
        for name, cls in [("deepfm", DeepFMRanker), ("din", DINRanker), ("esmm", ESMM)]:
            m = cls(cfg)
            sd = torch.load(DIR_CKPT / f"rank_{name}.pt", map_location="cpu", weights_only=False)
            m.load_state_dict(sd["model"])
            m.eval()
            self.models[name] = m
        self._lock = threading.Lock()

        # ================= 生产化运行栈（serving 包） =================
        # 护栏：质量门槛（i_score_decay，冷启豁免）+ 频控 + 已看/重复过滤
        q_stat = self.item_stat
        self.guardrails = Guardrails(
            min_quality=2.0,
            quality_of=lambda i: q_stat.get(int(i), {}).get("quality", 0.0),
            cold_items=self.cold_set,
            freq_cap=3, freq_window_s=600.0,
            max_exposure_rounds=3, exposure_dampen=0.5)
        self.exposure = ExposureHistory(window_requests=5)
        self._freq: dict[int, dict] = {}          # uid -> GuardContext.freq_window（跨请求持久）
        self.cand_cache = CandidateCache(capacity=512, ttl_s=120.0)

        # 反馈闭环：事件流落盘 data/feedback/events.jsonl（进程重启可回放）
        fb_dir = ROOT / "data" / "feedback"
        fb_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventLog(path=fb_dir / "events.jsonl")

        # 分层实验：rank_model 层 control=生产模型 / esmm / deepfm
        self.experiments = Experiments({
            "rank_model": {"salt": "L1-rank", "buckets": 100,
                           "arms": {"control": 50, "esmm": 25, "deepfm": 25}}})

        # 模型注册表：三个精排模型注册，din 经 staging 验证后首发上线
        self.registry = ModelRegistry(path=fb_dir / "registry.json")
        for nm, auc in [("din", 0.7295), ("esmm", 0.7194), ("deepfm", 0.7037)]:
            try:
                self.registry.register(
                    nm, weights=f"checkpoints/rank_{nm}.pt",
                    metrics={"auc_offline": auc})
            except Exception:  # 已注册（重启场景），幂等
                pass
        if self.registry.production() is None:
            self.registry.to_staging("din")
            self.registry.activate("din", note="服务首发上线")

        # 可观测性：延迟分位 + 分数漂移检测
        self.metrics = Registry()
        self.drift = DriftDetector(name="rank_score", baseline_n=100, window=100)
        self._started_at = time.time()

        log.info("启动完成: %.1fs（%d 用户 / %d 物品 / %d 冷启动物品）",
                 time.perf_counter() - t0, self.num_users, self.num_items, len(self.cold_set))

    # ---------------- 展示层 ----------------

    def _load_user_meta(self):
        """user_idx → {gender, age, occ} 展示标签（与 dataset.py 同口径反解）。"""
        raw = ROOT / "data" / "raw" / "users.dat"
        users_df = self.data.users_df.set_index("user_idx")
        out = []
        # ML-1M：users.dat 按行号即 user_id 排序，与 user_idx 构造一致
        if raw.exists():
            rows = [l.rstrip("\n").split("::") for l in raw.read_text(encoding="latin-1").splitlines()]
            # user_id, gender, age, occ, zip
            m = {int(r[0]): (r[1], r[2], r[3]) for r in rows}
        else:
            m = {}
        for i in range(self.num_users):
            g_i = int(users_df.loc[i, "gender_idx"]) if i in users_df.index else 0
            a_i = int(users_df.loc[i, "age_idx"]) if i in users_df.index else 0
            o_i = int(users_df.loc[i, "occ_idx"]) if i in users_df.index else 0
            # user_idx 由 sorted(user_id) 重编号，ML-1M 的 user_id 恰为 1..N 连续
            meta = m.get(i + 1, ("?", str(a_i), str(o_i)))
            out.append({
                "gender": GENDER_LABELS.get(g_i, "?"),
                "age": _age_label(a_i),
                "occ": OCC_LABELS[o_i] if o_i < len(OCC_LABELS) else "?",
            })
        return out

    def _item_card(self, i: int, score=None, cvr=None, why=None, cold=False) -> dict:
        s = self.item_stat.get(int(i), {})
        return {
            "item": int(i),
            "title": str(self.titles.get(int(i), "?")),
            "genres": str(self.genres.get(int(i), "")),
            "genre_list": str(self.genres.get(int(i), "")).split("|"),
            "score": None if score is None else float(score),
            "cvr": None if cvr is None else float(cvr),
            "why": why or [],
            "cold": cold or int(i) in self.cold_set,
            "n_users": s.get("n_users", 0),
            "mean_rating": round(s.get("mean_rating", 0.0), 2),
            "quality": round(s.get("quality", 0.0), 1),
        }

    # ---------------- API 实现 ----------------

    def bootstrap(self) -> dict:
        d = self.data
        cnt = d.users_df.set_index("user_idx")["u_cnt_total"].to_dict() if "u_cnt_total" in d.users_df else {}
        users = [{
            "id": i, "gender": self.user_meta[i]["gender"], "age": self.user_meta[i]["age"],
            "occ": self.user_meta[i]["occ"], "n": int(cnt.get(i, 0)),
        } for i in range(self.num_users)]
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_cold": len(self.cold_set),
            "models": list(self.models.keys()),
            # genre 下标从 1 开始（0 为序列 padding），按映射排序生成
            "genres": [self.gname[i] for i in sorted(self.gname)],
            "users": users,
        }

    def user_profile(self, uid: int) -> dict:
        d = self.data
        if not (0 <= uid < self.num_users):
            return {}
        hist_iids = [int(x) for x in d.seq_mat[uid].tolist() if x >= 0][-MAX_SEQ:]
        inter = d.interactions
        mine = inter[inter["user_idx"] == uid]
        ratings = dict(zip(mine["item_idx"], mine["rating"]))
        # 兴趣类目分布（按点击历史）
        gcount = {}
        for i in hist_iids:
            for g in str(self.genres.get(i, "")).split("|"):
                if g:
                    gcount[g] = gcount.get(g, 0) + 1
        total = sum(gcount.values()) or 1
        top_genres = sorted(gcount.items(), key=lambda x: -x[1])[:10]
        u = d.users_df.set_index("user_idx").loc[uid] if uid in d.users_df.set_index("user_idx").index else {}
        meta = self.user_meta[uid]
        return {
            "id": uid, "gender": meta["gender"], "age": meta["age"], "occ": meta["occ"],
            "n_ratings": int(u.get("u_cnt_total", len(mine))),
            "mean_rating": round(float(u.get("u_mean_rating", mine["rating"].mean())), 2),
            "active_months": int(u.get("u_active_months", 0)),
            "convert_rate": round(float(u.get("u_convert_rate", 0.0)), 3),
            "genre_dist": [{"name": g, "frac": round(c / total, 3)} for g, c in top_genres],
            "history": [{
                "item": int(i), "title": str(self.titles.get(int(i), "?")),
                "genres": str(self.genres.get(int(i), "")),
                "rating": int(ratings.get(int(i), 0)),
            } for i in reversed(hist_iids)],
        }

    def recommend(self, uid: int, model: str = "auto", topn: int = 10,
                  lam: float = 0.7, rerank: bool = True, cold_quota: int = 0) -> dict:
        d = self.data
        if not (0 <= uid < self.num_users):
            return {"error": "bad user"}
        topn = max(3, min(int(topn), 20))
        lam = max(0.0, min(float(lam), 1.0))
        cold_quota = max(0, min(int(cold_quota), 10))
        seen = set(int(x) for x in d.seq_mat[uid].tolist() if x >= 0)
        request_id = uuid.uuid4().hex[:12]

        # —— 实验分桶：auto 模式按哈希分桶定模型，control 走注册表生产模型
        if model == "auto":
            arm = self.experiments.assignment(uid, "rank_model")
            model = (self.registry.production() or "din") if arm == "control" else arm
        else:
            arm = f"manual:{model}"
            model = model if model in self.models else "din"

        # —— 召回三路（候选级缓存：同一用户 120s 内命中则跳过全部召回计算）——
        cache_hit = False
        t = {}
        cached = self.cand_cache.get(uid)
        if cached is not None:
            hot_ids, icf_ids, tt_ids, why, fused, t = cached
            cache_hit = True
        else:
            t0 = time.perf_counter()
            hot_ids = [i for i in self.hot.recall(topn=RECALL_TOPN, exclude=seen) if i not in seen]
            t["hot"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            hist = sorted(seen)   # ItemCF.recall 接收 list（`if not hist` 判空）
            icf_ids = [i for i in self.icf.recall(hist, topn=RECALL_TOPN) if i not in seen]
            t["icf"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            scores = self.item_emb @ self.user_emb[uid]
            scores[list(seen)] = -np.inf
            cand = np.argpartition(-scores, RECALL_TOPN)[: RECALL_TOPN]
            tt_ids = [int(i) for i in cand[np.argsort(-scores[cand])]]
            t["tt"] = (time.perf_counter() - t0) * 1000

            # —— 融合（去重合并，记录通道归因）——
            t0 = time.perf_counter()
            why = {}
            for ch, ids in [("热度", hot_ids[:20]), ("ItemCF", icf_ids[:30]), ("双塔", tt_ids[:40])]:
                for i in ids:
                    why.setdefault(int(i), []).append(ch)
            fused = list(why.keys())[:FUSE_TOPN]
            t["fuse"] = (time.perf_counter() - t0) * 1000
            self.cand_cache.put(
                uid, (hot_ids, icf_ids, tt_ids, why, fused, dict(t)),
                recall_cost_ms=t.get("hot", 0) + t.get("icf", 0) + t.get("tt", 0) + t.get("fuse", 0))

        # 冷启扶持：无行为物品按双塔分数注入
        cold_added = []
        if cold_quota > 0:
            cold_scores = {int(i): float(self.item_emb[i] @ self.user_emb[uid])
                           for i in self.cold_set if int(i) not in seen}
            cold_added = sorted(cold_scores, key=cold_scores.get, reverse=True)[:cold_quota]
            for i in cold_added:
                why.setdefault(i, []).append("冷启")
            fused = [i for i in fused if i not in cold_added] + cold_added

        # —— 安全护栏：频控 / 曝光饱和 / 重复 / 质量门槛（冷启豁免）——
        # 若拦得过狠（同用户高频刷新场景），自动降级：跳过频控/曝光规则保供给
        t0 = time.perf_counter()
        ctx = GuardContext(
            user_id=uid, seen=seen,
            recent_exposure=self.exposure.exposure_counts(uid),
            freq_window=self._freq.setdefault(uid, {}),
            now=time.time(), request_id=request_id)
        gr = self.guardrails.apply(fused, ctx)
        guardrail_relaxed = False
        if len(gr.passed) < max(5, topn):
            gr = self.guardrails.apply(
                fused, ctx, enable=("seen", "blocklist", "quality", "dedup"))
            guardrail_relaxed = True
        fused = gr.passed
        t["guardrail"] = (time.perf_counter() - t0) * 1000

        channels = {"热度": len(hot_ids), "ItemCF": len(icf_ids), "双塔": len(tt_ids),
                    "冷启": len(cold_added), "融合": len(fused)}

        if not fused:
            return {"error": "empty candidates"}

        # —— 精排 ——
        t0 = time.perf_counter()
        users = np.full(len(fused), uid, dtype=np.int64)
        items = np.asarray(fused, dtype=np.int64)
        batch = build_batch(d, users, items)
        with self._lock, torch.no_grad():
            out = self.models[model](batch)
            if model == "esmm":
                ctr, cvr = out[0], out[1]
            else:
                ctr, cvr = out, None
            preds = ctr.numpy()
            cvrs = cvr.numpy() if cvr is not None else None
        t["rank"] = (time.perf_counter() - t0) * 1000

        order = np.argsort(-preds)
        ranked = [fused[int(k)] for k in order]
        score_map = {fused[int(k)]: float(preds[int(k)]) for k in order}
        cvr_map = {fused[int(k)]: float(cvrs[int(k)]) for k in order} if cvrs is not None else {}

        # —— 重排（MMR：λ·相关性 - (1-λ)·内容相似度，贪心） ——
        t0 = time.perf_counter()
        before = ranked[:topn]

        def _cos(a: int, b: int) -> float:
            return float(np.dot(self.content_emb[a], self.content_emb[b]))

        if rerank and lam > 0:
            final = mmr_rerank(ranked[: topn * 3], score_map, _cos, topn=topn, lambda_div=lam)
        else:
            final = before

        # 冷启「扶持」= 保位曝光：注入的冷启动物品占据 Top-N 尾部席位
        # （与 07_eval_coldstart 的扶持额度语义一致：给无行为内容确定性流量）
        if cold_added:
            kept = [i for i in cold_added if i in score_map][: topn]
            final = [i for i in final if i not in kept][: max(0, topn - len(kept))] + kept
        t["rerank"] = (time.perf_counter() - t0) * 1000
        t["total"] = sum(t.values())

        # —— 反馈闭环：曝光事件入账 + 跨请求曝光记忆（防老面孔霸屏）——
        for pos, iid in enumerate(final):
            self.events.log(request_id, uid, int(iid), pos, "impression",
                            model=model, arm=arm, latency_ms=t["total"])
        self.exposure.record(uid, final)

        # —— 可观测性：延迟分位 + 分数漂移检测 ——
        self.metrics.incr("requests")
        for stage in ("hot", "icf", "tt", "fuse", "guardrail", "rank", "rerank"):
            if stage in t:
                self.metrics.observe_ms(stage, t[stage])
        self.metrics.observe_ms("total", t["total"])
        if final:
            self.drift.push(float(np.mean([score_map.get(i, 0.0) for i in final[:5]])))
            self.drift.check()

        def cards(ids):
            return [self._item_card(i, score=score_map.get(i), cvr=cvr_map.get(i),
                                    why=why.get(i, [])) for i in ids]

        def diversity(ids):
            if len(ids) < 2:
                return {"ild": 0.0, "coverage": 0.0}
            embs = self.content_emb[np.asarray(ids, dtype=np.int64)]
            embs = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-9)
            sim = embs @ embs.T
            n = len(ids)
            ild = float((1.0 - sim).sum() / (n * (n - 1)))
            gs = set()
            for i in ids:
                gs.update(str(self.genres.get(int(i), "")).split("|"))
            return {"ild": round(ild, 4), "coverage": round(len(gs) / self.vocab["num_genres"], 3)}

        return {
            "user": uid, "model": model, "topn": topn, "lam": lam,
            "rerank": bool(rerank), "cold_quota": cold_quota,
            "request_id": request_id, "arm": arm, "cache_hit": cache_hit,
            "guardrail_relaxed": guardrail_relaxed,
            "items": cards(final), "before": cards(before),
            "diversity": {"before": diversity(before), "after": diversity(final)},
            "channels": channels,
            "guardrail": {"blocked": gr.blocked_count, "trail": gr.trail(5)},
            "latency": {k: round(v, 2) for k, v in t.items()},
        }

    # ---------------- 生产运维 API ----------------

    def log_feedback(self, request_id: str, user: int, item: int,
                     pos: int, event: str) -> dict:
        """用户点击/点赞/点踩上报（幂等）。行为变化 → 候选缓存失效。"""
        ok = self.events.log(request_id, int(user), int(item), int(pos), event)
        if ok and event in ("click", "like", "dislike"):
            self.cand_cache.invalidate_user(int(user))
            self.metrics.incr(f"feedback_{event}")
        return {"logged": ok}

    def feedback_report(self) -> dict:
        return {
            "online_ctr": self.events.online_ctr(),
            "position_bias": self.events.position_bias(10),
            "snips": self.events.snips(),
            "arms": self.events.arm_report(),
            "log": self.events.stats(),
        }

    def ops_report(self) -> dict:
        m = self.metrics.snapshot()
        return {
            "uptime_s": round(time.time() - self._started_at, 1),
            "metrics": m,
            "cache": self.cand_cache.report(),
            "drift": self.drift.status(),
            "registry": {"production": self.registry.production(),
                         "models": self.registry.list()},
            "registry_history": self.registry.history()[-6:],
            "feedback": self.events.stats(),
        }

    def experiment_info(self, uid: int) -> dict:
        sample = list(range(0, self.num_users, 7))
        return {
            "layer": "rank_model",
            "arms": {"control": 50, "esmm": 25, "deepfm": 25},
            "your_arm": self.experiments.assignment(uid, "rank_model"),
            "your_model": (self.registry.production() or "din"),
            "traffic_split": self.experiments.layer_report("rank_model", sample),
        }

    def registry_activate(self, name: str) -> dict:
        """上线指定模型（须先过 staging 影子验证 → activate 自动流转）。"""
        self.registry.to_staging(name)
        self.registry.activate(name, note="web 控制台切换")
        return {"production": self.registry.production(),
                "models": self.registry.list()}

    def similar(self, iid: int, k: int = 8) -> dict:
        if not (0 <= iid < self.num_items):
            return {"error": "bad item"}
        emb = self.content_emb / np.maximum(
            np.linalg.norm(self.content_emb, axis=1, keepdims=True), 1e-9)
        sims = emb @ emb[iid]
        sims[iid] = -1
        top = np.argsort(-sims)[:k]
        return {"item": iid, "items": [
            self._item_card(int(i), score=float(sims[int(i)]), why=["内容相似"]) for i in top]}

    def item_detail(self, iid: int) -> dict:
        if not (0 <= iid < self.num_items):
            return {"error": "bad item"}
        d = self.data
        pop_rank = int((d.item_counts > d.item_counts[iid]).sum()) + 1
        card = self._item_card(iid)
        card["pop_rank"] = pop_rank
        card["n_inter"] = int(d.item_counts[iid])
        return card


STATE: RecSysState | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静默默认访问日志
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str):
        try:
            body = path.read_bytes()
        except OSError:
            self._json({"error": "not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/" or u.path == "/index.html":
                self._file(STATIC / "index.html", "text/html; charset=utf-8")
            elif u.path == "/api/bootstrap":
                self._json(STATE.bootstrap())
            elif u.path.startswith("/api/user/"):
                self._json(STATE.user_profile(int(u.path.rsplit("/", 1)[1])))
            elif u.path == "/api/recommend":
                self._json(STATE.recommend(
                    uid=int(q.get("user", 7)), model=q.get("model", "auto"),
                    topn=int(q.get("topn", 10)), lam=float(q.get("lam", 0.7)),
                    rerank=q.get("rerank", "1") not in ("0", "false"), cold_quota=int(q.get("cold", 0))))
            elif u.path.startswith("/api/item/"):
                parts = [p for p in u.path.split("/") if p]
                iid = int(parts[-1] if not parts[-1] == "similar" else parts[-2])
                if parts[-1] == "similar":
                    self._json(STATE.similar(iid, k=min(int(q.get("k", 8)), 12)))
                else:
                    self._json(STATE.item_detail(iid))
            elif u.path == "/api/ops":
                self._json(STATE.ops_report())
            elif u.path == "/api/ops/prometheus":
                body = STATE.metrics.expose_prometheus().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/experiment":
                self._json(STATE.experiment_info(int(q.get("user", 0))))
            elif u.path == "/api/feedback/report":
                self._json(STATE.feedback_report())
            elif u.path == "/api/health":
                self._json({"status": "ok" if STATE else "starting",
                            "uptime_s": round(time.time() - STATE._started_at, 1),
                            "drift": STATE.drift.status()["drift"],
                            "models_loaded": list(STATE.models)})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            log.exception("API %s 失败", u.path)
            STATE.metrics.incr("errors")
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/feedback":
                self._json(STATE.log_feedback(
                    body.get("request_id", ""), int(body.get("user", 0)),
                    int(body.get("item", 0)), int(body.get("pos", 0)),
                    str(body.get("event", "click"))))
            elif u.path == "/api/registry/activate":
                self._json(STATE.registry_activate(str(body.get("name", ""))))
            elif u.path == "/api/registry/rollback":
                name = STATE.registry.rollback()
                self._json({"rolled_back_to": name,
                            "production": STATE.registry.production()})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            log.exception("POST %s 失败", u.path)
            STATE.metrics.incr("errors")
            self._json({"error": str(e)}, 500)


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    log.info("正在预热（加载数据 / 训练 ItemCF / 加载 3 个精排模型）……")
    STATE = RecSysState()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("Web Demo 就绪: http://%s:%d/", args.host, args.port)
    print(f"\n  🎬 UGC 推荐系统 Web Demo → http://{args.host}:{args.port}/\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
