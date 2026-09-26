"""向量召回索引：暴力 baseline vs Faiss IVF，对比召回率与吞吐。

工程上关心的是「近似检索丢了多少召回」以及「换来多少 QPS」，因此这里把两者都实现了，
并在 README 的实验表里给出 trade-off 实测数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import faiss
import numpy as np


@dataclass
class IndexReport:
    metric: str
    recall_vs_flat: float
    build_sec: float
    qps: float
    ntotal: int


class ANNIndex:
    """统一封装 Flat / IVF / HNSW，暴露相同的 `search` 接口便于做对照实验。"""

    def __init__(self, vectors: np.ndarray, metric: str = "ip"):
        """
        Args:
            vectors: (N, D) float32，建议已做 L2 归一化以便内积等价于余弦。
            metric: 'ip' 内积（余弦）或 'l2' 欧氏距离。
        """
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.dim = self.vectors.shape[1]
        self.metric = metric
        self._flat = faiss.IndexFlatIP(self.dim) if metric == "ip" else faiss.IndexFlatL2(self.dim)
        self._flat.add(self.vectors)
        self._index = None
        self._kind = "flat"

    # -------------------------------------------------------------- 建索引
    def build_flat(self) -> "ANNIndex":
        self._index = self._flat
        self._kind = "flat"
        return self

    def build_ivf(self, nlist: int = 100, nprobe: int = 10, train_sub: int = 200_000) -> "ANNIndex":
        quantizer = (
            faiss.IndexFlatIP(self.dim) if self.metric == "ip" else faiss.IndexFlatL2(self.dim)
        )
        factory = f"IVF{nlist},Flat"
        idx = faiss.index_factory(self.dim, factory,
                                  faiss.METRIC_INNER_PRODUCT if self.metric == "ip"
                                  else faiss.METRIC_L2)
        train_x = self.vectors[: min(train_sub, len(self.vectors))]
        idx.train(train_x)
        idx.add(self.vectors)
        idx.nprobe = nprobe
        self._index = idx
        self._kind = f"ivf{nlist}_probe{nprobe}"
        return self

    def build_hnsw(self, m: int = 32, ef_search: int = 64) -> "ANNIndex":
        idx = faiss.IndexHNSWFlat(self.dim, m,
                                  faiss.METRIC_INNER_PRODUCT if self.metric == "ip"
                                  else faiss.METRIC_L2)
        idx.hnsw.efSearch = ef_search
        idx.add(self.vectors)
        self._index = idx
        self._kind = f"hnsw_m{m}_ef{ef_search}"
        return self

    # -------------------------------------------------------------- 检索
    def search(self, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        scores, ids = self._index.search(q, topk)
        return scores, ids

    def search_flat(self, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        return self._flat.search(q, topk)

    @property
    def kind(self) -> str:
        return self._kind


def recall_overlap(approx_ids: np.ndarray, gt_ids: np.ndarray) -> float:
    """近似检索结果对暴力检索结果的保持率（Recall@K vs Flat@K）。"""
    assert approx_ids.shape == gt_ids.shape
    hits = 0
    total = 0
    for a, g in zip(approx_ids, gt_ids):
        gs = set(g.tolist())
        total += len(gs)
        hits += len(set(a.tolist()) & gs)
    return hits / max(total, 1)


def benchmark(index: ANNIndex, queries: np.ndarray, topk: int = 50) -> Dict[str, float]:
    """返回该索引相对暴力检索的召回保持率 + QPS 实测。"""
    import time

    qs = np.ascontiguousarray(queries, dtype=np.float32)
    flat_ids = index.search_flat(qs, topk)[1]

    t0 = time.perf_counter()
    approx_ids = index.search(qs, topk)[1]
    cost = time.perf_counter() - t0

    return {
        "kind": index.kind,  # type: ignore[dict-item]
        "recall_vs_flat": float(recall_overlap(approx_ids, flat_ids)),
        "qps": float(len(qs) / max(cost, 1e-9)),
        "latency_ms_per_q": float(cost * 1000 / max(len(qs), 1)),
    }
