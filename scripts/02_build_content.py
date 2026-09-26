#!/usr/bin/env python
"""Step 2：内容理解 —— 语义向量 + 零样本标签生成。

产出 data/processed/content/{content_item_idx.npy, content_emb.npy, content_tags.json}，

用法：
    python scripts/02_build_content.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.coldstart.content import (  # noqa: E402
    ItemContent,
    build_item_text,
    encode_texts,
    zero_shot_tags,
)
from recsys.common import DIR_PROC, get_logger, save_json, timer  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "bge-small-en-v1.5"

# 受控标签词表：类目词 + 内容向描述词。用可控词表而非开放生成，
# 是为了让标签能直接进倒排索引 / 召回通道，而不是只停留在演示层面。
TAG_VOCAB = [
    "action", "adventure", "animation", "comedy", "crime", "documentary", "drama",
    "family", "fantasy", "film-noir", "horror", "musical", "mystery", "romance",
    "sci-fi", "thriller", "war", "western",
    "fast-paced storytelling", "slow-burn character study", "visual spectacle",
    "plot twist driven", "coming of age", "dark and gritty atmosphere",
    "lighthearted and humorous", "emotional and tear-jerking",
    "thought-provoking themes", "based on real events", "ensemble cast",
    "for children and families", "suspense and tension", "mind-bending structure",
    "romantic relationship focus", "crime investigation", "war and battlefield",
    "supernatural elements", "historical period drama", "musical performance",
    "cult classic", "arthouse cinema", "high concept premise", "franchise sequel",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--top_k_tags", type=int, default=3)
    p.add_argument("--tag_threshold", type=float, default=0.32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("content")
    proc = Path(args.proc_dir)
    items = pd.read_csv(proc / "items.csv")
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = [
        build_item_text(t, g)
        for t, g in zip(items["clean_title"].tolist(), items["genres"].tolist())
    ]
    logger.info("待编码物品 %d 条，设备=%s", len(texts), device)
    with timer("语义向量编码", logger):
        emb = encode_texts(texts, args.model_path, batch_size=args.batch_size, device=device)
    logger.info("向量维度: %s", emb.shape)

    with timer("标签向量编码", logger):
        tag_emb = encode_texts(TAG_VOCAB, args.model_path, batch_size=32, device=device)
    tags_per_item = zero_shot_tags(
        emb, tag_emb, TAG_VOCAB, top_k=args.top_k_tags, threshold=args.tag_threshold
    )

    content = ItemContent(
        item_idx=items["item_idx"].to_numpy(dtype=np.int64),
        emb=emb,
        tags={int(i): t for i, t in zip(items["item_idx"].tolist(), tags_per_item)},
    )
    content.save(proc / "content")

    n_tagged = sum(1 for v in content.tags.values() if v)
    stat = {
        "model": Path(args.model_path).name,
        "dim": int(emb.shape[1]),
        "num_items": int(len(items)),
        "tag_vocab_size": len(TAG_VOCAB),
        "items_with_tag": int(n_tagged),
        "tag_coverage": float(n_tagged / max(len(items), 1)),
        "avg_tags_per_item": float(np.mean([len(v) for v in content.tags.values()])),
    }
    save_json(stat, proc / "content" / "content_stat.json")
    examples = [
        {"title": items.loc[i, "clean_title"], "tags": content.tags[int(items.loc[i, "item_idx"])]}
        for i in range(0, 60, 12)
    ]
    save_json(examples, proc / "content" / "tag_examples.json")
    logger.info("内容资产落盘完成: %s", stat)
    print(json.dumps({"stat": stat, "examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
