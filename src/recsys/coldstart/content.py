"""冷启动与内容理解：语义向量、语义标签生成、零行为物品召回。

冷启动的本质问题是「没有协同信号时靠什么排序」。这里用内容语义向量替代 ID embedding：
1) 用 bge-small-en-v1.5 把标题 / 类目编码成 384 维语义向量；
2) 用同样向量空间做零样本标签生成（候选标签与物品相似度打分），把内容理解结果
   变成可被召回侧直接使用的结构化标签；
3) 冷启动物品向量 = 内容塔输出（不含 ID embedding），用户侧用历史点击均值构造画像。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch


def build_item_text(title: str, genres: str) -> str:
    """把元信息拼成一句自然语言，作为语义模型的输入。"""
    g = genres.replace("|", ", ")
    return f"{title}. genres: {g}."


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware 均值池化：padding 位置不参与平均（否则短文本向量会被稀释）。"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-8)
    return summed / counts


def encode_texts(texts: Sequence[str], model_path: str | Path, batch_size: int = 64,
                 device: str = "cpu", normalize: bool = True, max_length: int = 128) -> np.ndarray:
    """离线编码：直接走 transformers + 手写 pooling。

    不依赖 sentence-transformers 的 modules.json 组合路径，因此不受其版本间
    pooling 配置格式变化的影响，只依赖权重本体。
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModel.from_pretrained(str(model_path))
    model.to(device).eval()

    out_batches = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = [str(t) for t in texts[start : start + batch_size]]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            pooled = _mean_pool(hidden, enc["attention_mask"])
            out_batches.append(pooled.float().cpu().numpy())
    emb = np.concatenate(out_batches, axis=0).astype(np.float32)
    if normalize:
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb


@dataclass
class ItemContent:
    """物品侧内容资产：语义向量 + 自动标签。"""

    item_idx: np.ndarray
    emb: np.ndarray
    tags: Dict[int, List[str]] = field(default_factory=dict)

    def save(self, dirpath: str | Path) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "content_item_idx.npy", self.item_idx)
        np.save(d / "content_emb.npy", self.emb)
        (d / "content_tags.json").write_text(
            json.dumps({str(k): v for k, v in self.tags.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dirpath: str | Path) -> "ItemContent":
        d = Path(dirpath)
        return cls(
            item_idx=np.load(d / "content_item_idx.npy"),
            emb=np.load(d / "content_emb.npy"),
            tags={int(k): v for k, v in json.loads(
                (d / "content_tags.json").read_text(encoding="utf-8")).items()},
        )


def zero_shot_tags(
    emb_items: np.ndarray,
    emb_tags: np.ndarray,
    tag_names: Sequence[str],
    top_k: int = 3,
    threshold: float = 0.3,
) -> List[List[str]]:
    """标签生成：候选标签与物品在同一向量空间里算余弦，取 top-k。

    相比让 LLM 直接输出词表外词汇，这种做法天然保证标签可控、可入库、可复现；
    若有在线 LLM（OpenAI 兼容接口）可再做一轮标签润色，属于可插拔增强。
    """
    sims = emb_items @ emb_tags.T  # (N, T)
    out: List[List[str]] = []
    for row in sims:
        idx = np.argsort(-row)[:top_k]
        out.append([tag_names[i] for i in idx if row[i] >= threshold])
    return out


class ContentRecall:
    """零行为物品的内容召回：物品向量直接检索，用历史均值画像构造用户向量。"""

    def __init__(self, item_idx: np.ndarray, item_emb: np.ndarray):
        self.item_idx = np.asarray(item_idx, dtype=np.int64)
        self.item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        self.norm_emb = self.item_emb / (
            np.linalg.norm(self.item_emb, axis=1, keepdims=True) + 1e-8
        )

    def user_vec_from_history(self, hist_ids: Sequence[int]) -> np.ndarray | None:
        """用户画像 = 历史点击物品内容向量的均值（也可换成 attended 版本）。"""
        pos = {int(i): k for k, i in enumerate(self.item_idx)}
        rows = [pos[h] for h in hist_ids if h in pos]
        if not rows:
            return None
        v = self.item_emb[rows].mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-8)

    def recall(self, user_vec: np.ndarray, topn: int = 50,
               candidate_items: Sequence[int] | None = None) -> List[int]:
        if candidate_items is None:
            sims = self.norm_emb @ user_vec
            cand = np.argpartition(-sims, min(topn, len(sims)))[:topn]
            return self.item_idx[cand[np.argsort(-sims[cand])]].tolist()
        pos = {int(i): k for k, i in enumerate(self.item_idx)}
        rows = np.array([pos[c] for c in candidate_items if c in pos], dtype=np.int64)
        sims = self.norm_emb[rows] @ user_vec
        order = rows[np.argsort(-sims)[:topn]]
        return self.item_idx[order].tolist()


def cold_or_warm_mask(train_item_counts: np.ndarray, warm_threshold: int = 5) -> np.ndarray:
    """返回 True=冷门（训练集中正向行为不足 warm_threshold 条）。"""
    return np.asarray(train_item_counts) < warm_threshold


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
