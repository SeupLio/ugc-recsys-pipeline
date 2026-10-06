"""流式在线学习（Streaming / Online Learning）—— 真实事件流上的先序评测。

传统离线评测的两个「不真实」：随机切分（用未来训练、用过去测试），
以及一次性训练（模型永不更新）。本模块把 ML-1M 的 98.8 万条真实评分
按时间戳排成事件流，做「先序评测（prequential / test-then-train）」：

    for 每个时间窗 W（按真实时间顺序）:
        1. 用「当前模型」给 W 打分（模型还没见过 W —— 无泄漏）
        2. 记录 AUC / 分数分布（PSI 漂移）
        3. partial_fit(W)：增量更新模型
        4. 更新时间衰减热度（在线组件）

组件：
- load_event_stream   ：interactions.csv → 按时间排序的事件数组
- DecayedPopularity   ：连续时间衰减热度（在线可更新，替代离线静态热榜）
- OnlineRanker        ：精排模型的增量 SGD 包装（partial_fit / score_batch）
- PrequentialReplayer ：编排整个回放，产出逐窗指标轨迹

真实流量的价值：ML-1M 的时间戳带着真实特性——流行度漂移（大片下映）、
活跃用户突发（周末效应）、长尾行为稀疏。这里所有指标都在真实时间轴上
先序计算，比随机切分的离线 AUC 更接近线上。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .metrics import DriftDetector


# ---------------- 事件流 ----------------

def load_event_stream(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """interactions.csv → (users, items, ts)，全量按时间升序。

    返回索引化的 user/item（与 processed 体系一致）。真实行为流。
    """
    df = pd.read_csv(path, usecols=["user_idx", "item_idx", "ts"])
    df = df.sort_values("ts", kind="stable")
    return (df.user_idx.to_numpy(np.int64),
            df.item_idx.to_numpy(np.int64),
            df.ts.to_numpy(np.int64))


class UserHistory:
    """随流维护的用户交互历史（负采样排除 + 已看过滤）。"""

    def __init__(self, num_users: int) -> None:
        self.seen: List[set] = [set() for _ in range(num_users)]

    def add(self, users: np.ndarray, items: np.ndarray) -> None:
        for u, i in zip(users.tolist(), items.tolist()):
            self.seen[u].add(i)

    def sample_negatives(self, users: np.ndarray, rng: np.random.Generator,
                         num_items: int, pop_p: Optional[np.ndarray] = None) -> np.ndarray:
        """每个正样本采一个未看过的负样本（热度加权，贴近真实曝光分布）。

        向量化：先按分布批量抽，再对「已看过」的逐个拒绝重抽（均匀）。
        """
        if pop_p is not None and pop_p.sum() > 0:
            cum = np.cumsum(pop_p)
            cands = np.searchsorted(cum, rng.random(len(users)))
        else:
            cands = rng.integers(0, num_items, len(users))
        out = cands.astype(np.int64)
        for k, u in enumerate(users.tolist()):
            if out[k] not in self.seen[u]:
                continue
            for _ in range(8):                      # 拒绝采样，8 次几乎必中
                cand = int(rng.integers(num_items))
                if cand not in self.seen[u]:
                    out[k] = cand
                    break
        return out


# ---------------- 在线热度 ----------------

class DecayedPopularity:
    """连续时间衰减热度榜（half-life 语义，随事件流在线更新）。"""

    def __init__(self, num_items: int, half_life_s: float = 30 * 86400.0,
                 eps: float = 1e-6) -> None:
        self.counts = np.full(num_items, eps, dtype=np.float64)
        self._last_ts: Optional[float] = None
        self._hl = float(half_life_s)
        self.top_cache: Optional[np.ndarray] = None

    def advance(self, ts: float) -> None:
        """把内部时钟推进到 ts（先衰减再记账，保证时间一致性）。"""
        if self._last_ts is None:
            self._last_ts = ts
            return
        dt = max(0.0, ts - self._last_ts)
        self.counts *= 0.5 ** (dt / self._hl)
        self._last_ts = ts
        self.top_cache = None

    def add(self, ts: float, items: np.ndarray) -> None:
        self.advance(ts)
        np.add.at(self.counts, items, 1.0)
        self.top_cache = None

    def top(self, k: int) -> np.ndarray:
        if self.top_cache is None or len(self.top_cache) < k:
            self.top_cache = np.argsort(-self.counts)
        return self.top_cache[:k]

    def probs(self) -> np.ndarray:
        """热度加权负采样分布（√平滑防头部垄断）。"""
        w = np.sqrt(self.counts)
        return w / w.sum()

    def churn(self, k: int = 100) -> Dict[str, float]:
        """当前 Top-k 热度统计（用于观察榜单换血）。"""
        t = self.top(k)
        return {"top_k": k, "entropy": float(-np.sum(
            (p := self.counts[t] / self.counts.sum()) * np.log(p + 1e-12))),
            "gini": float(np.abs(np.cumsum(np.sort(self.counts)) /
                                 self.counts.sum() -
                                 np.linspace(0, 1, len(self.counts))).mean() * 2)}


# ---------------- 在线精排 ----------------

class OnlineRanker:
    """精排模型增量更新包装（partial_fit）。保持 torch 模型引用不复制。

    与离线训练器（05）同口径用 Adam + weight_decay；差异是「一次只吃
    一个时间窗」的 partial_fit 与先序评测时的 no_grad 打分。
    """

    def __init__(self, model, lr: float = 1e-3, wd: float = 1e-6) -> None:
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        self.n_updates = 0

    @torch.no_grad()
    def score(self, batch) -> np.ndarray:
        out = self.model(batch)
        ctr = out[0] if isinstance(out, (tuple, list)) else out
        return ctr.numpy()

    def partial_fit(self, batch, labels: np.ndarray, n_steps: int = 1,
                    minibatch: int = 2048) -> float:
        """在窗口数据上做 n_steps 轮 SGD（内部切 minibatch）。返回最后一步 loss。

        模型输出是 sigmoid 概率，故对概率做 BCE（clamp 防数值边界）。
        """
        y_all = torch.as_tensor(labels, dtype=torch.float32)
        n = y_all.shape[0]
        mb = max(256, min(int(minibatch), n))
        loss_v = 0.0
        for _ in range(n_steps):
            perm = torch.randperm(n)
            for s in range(0, n, mb):
                idx = perm[s: s + mb]
                sub = {k: v[idx] for k, v in batch.items()}
                y = y_all[idx]
                out = self.model(sub)
                ctr = out[0] if isinstance(out, (tuple, list)) else out
                loss = torch.nn.functional.binary_cross_entropy(
                    ctr.clamp(1e-6, 1 - 1e-6), y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                loss_v = float(loss.detach())
                self.n_updates += 1
        return loss_v


# ---------------- 先序回放编排 ----------------

@dataclass
class WindowReport:
    idx: int
    ts_start: int
    n_events: int
    auc: float
    auc_pop: float              # 热度基线（对照组）
    mean_pos: float
    mean_neg: float
    psi: float
    drift: bool
    loss: float
    top_churn: float            # 与上一窗 Top100 的换血率

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


class PrequentialReplayer:
    """test-then-train 主循环。

    warmup_frac 之前的事件只用于热身训练（不参与评测）；之后逐窗：
    先用当前模型打分（先序，无泄漏），再增量更新模型与热度。
    """

    def __init__(self, model, num_items: int, num_users: int,
                 window: int = 20_000, lr: float = 1e-3,
                 half_life_s: float = 30 * 86400.0, seed: int = 42) -> None:
        self.ranker = OnlineRanker(model, lr=lr)
        self.pop = DecayedPopularity(num_items, half_life_s=half_life_s)
        self.hist = UserHistory(num_users)
        self.window = int(window)
        self.rng = np.random.default_rng(seed)

    def auc(self, pos: np.ndarray, neg: np.ndarray) -> float:
        """手写 AUC（秩统计，与 eval/metrics 口径一致）。"""
        from recsys.eval.metrics import auc as _auc  # 复用主链路实现
        return float(_auc(np.concatenate([np.zeros(len(neg)), np.ones(len(pos))]),
                          np.concatenate([neg, pos])))

    def run(self, users: np.ndarray, items: np.ndarray, ts: np.ndarray,
            build_batch, warmup_frac: float = 0.8,
            on_window=None) -> List[WindowReport]:
        n = len(users)
        cut = int(n * warmup_frac)
        reports: List[WindowReport] = []
        drift = DriftDetector(name="score", baseline_n=20_000, window=20_000)
        prev_top: Optional[set] = None
        # 热度与历史在 warmup 段先行更新（模型还没上线时的「真实历史」）
        self.pop.add(float(ts[0]), items[:1])
        for s in range(0, cut, self.window):
            e = min(s + self.window, cut)
            self.pop.advance(float(ts[e - 1]))
            np.add.at(self.pop.counts, items[s:e], 1.0)
            self.pop.top_cache = None
        self.hist.add(users[:cut], items[:cut])

        for s in range(cut, n, self.window):
            e = min(s + self.window, n)
            u_w, i_w, t_w = users[s:e], items[s:e], ts[s:e]
            neg = self.hist.sample_negatives(
                u_w, self.rng, self.pop.counts.shape[0], pop_p=self.pop.probs())

            # 1) 先序打分（模型未见过本窗数据）
            b_pos = build_batch(u_w, i_w)
            b_neg = build_batch(u_w, neg)
            p_pos = self.ranker.score(b_pos)
            p_neg = self.ranker.score(b_neg)
            p_pop = self.pop.counts[i_w]             # 热度基线分
            n_pop = self.pop.counts[neg]

            auc_m = self.auc(p_pos, p_neg)
            auc_b = self.auc(p_pop, n_pop)
            drift.push(float(p_pos.mean()))
            psi = drift.check()

            # 2) test-then-train：增量更新（模型 + 热度 + 历史）
            labels = np.concatenate([np.ones(len(u_w)), np.zeros(len(u_w))])
            loss = self.ranker.partial_fit(
                build_batch(np.concatenate([u_w, u_w]),
                            np.concatenate([i_w, neg])), labels)
            self.pop.add(float(t_w[0]), i_w)
            self.hist.add(u_w, i_w)

            cur_top = set(self.pop.top(100).tolist())
            churn = 1.0 - len(cur_top & prev_top) / 100 if prev_top else 1.0
            prev_top = cur_top

            rep = WindowReport(
                idx=len(reports), ts_start=int(t_w[0]), n_events=e - s,
                auc=auc_m, auc_pop=auc_b,
                mean_pos=float(p_pos.mean()), mean_neg=float(p_neg.mean()),
                psi=round(psi, 4), drift=bool(drift.drift), loss=round(loss, 4),
                top_churn=round(churn, 3))
            reports.append(rep)
            if on_window:
                on_window(rep)
        return reports
