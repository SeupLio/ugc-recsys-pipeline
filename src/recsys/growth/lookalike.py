"""用户增长模块：相似人群扩展（Lookalike）与用户价值预估（pLTV）。

对应 JD 里「目标用户识别、相似人群扩展、用户价值预估」这条职责：
- Lookalike：把种子用户的 Embedding 作为锚点，在双塔用户向量上做 KNN 扩散；
- pLTV：用画像 + Embedding 回归用户未来价值，按价值分层指导投放。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


class Lookalike:
    """基于用户向量的相似人群扩展。

    支持两种扩散方式：
    - centroid：种子集质心外扩，召回率高但容易跑偏到大众人群；
    - knn：每个种子各自取近邻后按命中次数投票，精度高、可控性强。
    """

    def __init__(self, user_emb: np.ndarray):
        self.user_emb = np.ascontiguousarray(user_emb, dtype=np.float32)
        norms = np.linalg.norm(self.user_emb, axis=1, keepdims=True)
        self.norm_emb = self.user_emb / np.clip(norms, 1e-8, None)

    def expand_centroid(self, seeds: Sequence[int], candidate_pool: Sequence[int],
                        ratio: float = 0.1) -> List[int]:
        pool = np.asarray(list(candidate_pool), dtype=np.int64)
        centroid = self.user_emb[np.asarray(list(seeds), dtype=np.int64)].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = self.norm_emb[pool] @ centroid
        k = max(1, int(len(pool) * ratio))
        top = np.argpartition(-sims, k)[:k]
        return pool[top[np.argsort(-sims[top])]].tolist()

    def expand_knn(self, seeds: Sequence[int], candidate_pool: Sequence[int],
                   topk_per_seed: int = 20, ratio: float = 0.1) -> List[int]:
        pool = np.asarray(list(candidate_pool), dtype=np.int64)
        seed_set = set(int(s) for s in seeds)
        pool = np.array([u for u in pool if u not in seed_set], dtype=np.int64)
        votes: Dict[int, float] = {}
        for s in seeds:
            sims = self.norm_emb[pool] @ self.norm_emb[int(s)]
            k = min(topk_per_seed, len(pool))
            top = np.argpartition(-sims, k)[:k]
            for u, sc in zip(pool[top], sims[top]):
                votes[int(u)] = votes.get(int(u), 0.0) + float(max(sc, 0.0))
        ordered = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        k = max(1, int(len(ordered) * ratio))
        return [u for u, _ in ordered[:k]]


@dataclass
class ValueModelConfig:
    hidden_ratio: float = 0.5
    max_iter: int = 400
    seed: int = 42


class UserValueModel:
    """用户价值预估：梯度提升树 +手工解释性。

    用 sklearn HistGradientBoostingRegressor 而不是黑箱大模型，原因是增长场景里
    「能否向业务解释谁是高价值用户」与 AUC 同等重要。
    """

    def __init__(self, cfg: ValueModelConfig | None = None):
        self.cfg = cfg or ValueModelConfig()
        self.model = None
        self.feature_names: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str] | None = None) -> "UserValueModel":
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.feature_names = list(feature_names) if feature_names else [
            f"f{i}" for i in range(X.shape[1])
        ]
        self.model = HistGradientBoostingRegressor(
            max_iter=self.cfg.max_iter, random_state=self.cfg.seed, learning_rate=0.06
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("请先调用 fit()")
        return self.model.predict(X)

    def top_feature_importance(self, X: np.ndarray, y: np.ndarray, k: int = 5
                               ) -> List[Tuple[str, float]]:
        """用 permutation importance 给出可解释的特征排序。"""
        if self.model is None:
            return []
        from sklearn.inspection import permutation_importance

        r = permutation_importance(
            self.model, X, y, n_repeats=3, random_state=self.cfg.seed, scoring="neg_mean_squared_error"
        )
        order = np.argsort(-r.importances_mean)[:k]
        return [(self.feature_names[i], float(r.importances_mean[i])) for i in order]


def build_value_features(user_df, user_emb: np.ndarray, emb_dim: int = 32) -> Tuple[np.ndarray, List[str]]:
    """把画像统计 + 用户向量降维后拼成价值模型的特征矩阵。"""
    num_cols = [
        "u_cnt_total", "u_cnt_click", "u_cnt_convert", "u_mean_rating",
        "u_convert_rate", "u_click_rate", "u_recent10_mean_rating", "u_active_months",
    ]
    X_num = user_df[num_cols].to_numpy(dtype=np.float32)
    X_num = np.nan_to_num(X_num, nan=0.0)

    emb = user_emb[: X_num.shape[0]]
    if emb.shape[1] > emb_dim:
        # 用 PCA 压缩保留主要用户兴趣方向，避免高维噪声淹没统计特征
        from sklearn.decomposition import PCA

        pca = PCA(n_components=emb_dim, random_state=42)
        emb = pca.fit_transform(emb).astype(np.float32)
    X = np.concatenate([X_num, emb], axis=1)
    names = list(num_cols) + [f"emb_{i}" for i in range(emb.shape[1])]
    return X, names
