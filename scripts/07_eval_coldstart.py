#!/usr/bin/env python
"""Step 7：冷启动（新内容分发）评测。

业务问题：一条全新的内容没有任何行为数据，协同模型学不到它的 ID Embedding，
于是它永远排不进候选项 —— 这就是「冷启动饿死」。
这里复现工业界的标准解法：给新内容开一条独立通道并按额度混入召回候选。

对比通道：
1) 纯双塔（无冷启通道）：冷启动内容的理论下限；
2) 内容语义通道：用户历史的内容向量均值画像 → 检索新内容；
3) 标签倒排通道：把 Step2 生成的语义标签当倒排键做召回；
配额 M 表示候选里有几个坑位留给新内容，这对应到线上的「新内容扶持额度」。

用法：
    python scripts/07_eval_coldstart.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.coldstart.content import ContentRecall  # noqa: E402
from recsys.common import DIR_PROC, DIR_RES, get_logger, save_json  # noqa: E402
from recsys.data.torch_data import load_processed  # noqa: E402
from recsys.eval.metrics import evaluate_topk, hit_rate_at_k  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", type=str, default=str(DIR_PROC))
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--quotas", type=int, nargs="+", default=[5, 10, 20])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("coldstart")
    proc = Path(args.proc_dir)
    data = load_processed(proc)

    cold_items = np.load(proc / "cold_items.npy")
    cold_pairs = pd.read_csv(proc / "cold_pairs.csv")
    user_emb = np.load(proc / "user_emb.npy")
    item_emb = np.load(proc / "item_emb.npy")

    content = np.load(proc / "content" / "content_emb.npy")
    content_idx = np.load(proc / "content" / "content_item_idx.npy")
    tags = json.loads((proc / "content" / "content_tags.json").read_text(encoding="utf-8"))
    content_rec = ContentRecall(content_idx, content)

    cold_set = set(int(i) for i in cold_items)
    u2gt: dict[int, list[int]] = defaultdict(list)
    for _, r in cold_pairs.iterrows():
        u2gt[int(r["user_idx"])].append(int(r["item_idx"]))
    users = sorted(u2gt)
    logger.info("冷启动评测用户 %d，冷启动物品 %d，评测对 %d",
                len(users), len(cold_items), len(cold_pairs))

    hist = {u: [int(i) for i in data.seq_mat[u][data.seq_mat[u] >= 0].numpy()] for u in users}

    # ---------------------------------------------------- 通道 1：纯双塔基线
    base_pred = {}
    for u in users:
        scores = item_emb @ user_emb[u]
        seen = set(hist[u]) | set(u2gt[u])
        scores[list(seen)] = -np.inf
        cand = np.argpartition(-scores, args.topk)[: args.topk]
        base_pred[u] = cand[np.argsort(-scores[cand])].tolist()

    # -------------------------------------- 通道 2：内容语义（用户画像 → 新内容）
    content_pred = {}
    for u in users:
        uv = content_rec.user_vec_from_history(hist[u])
        if uv is None:
            content_pred[u] = []
            continue
        content_pred[u] = [int(i) for i in content_rec.recall(uv, topn=args.topk,
                                                             candidate_items=list(cold_set))]

    # --------------------------------------------- 通道 3：标签倒排（AQ-TF 打分）
    inv = defaultdict(list)
    for i in cold_set:
        for t in tags.get(str(int(i)), []):
            inv[t].append(int(i))
    tag_pred = {}
    for u in users:
        profile = Counter()
        for j, item in enumerate(hist[u][-30:]):
            w = 1.0 / np.log2(j + 2)
            for t in tags.get(str(int(item)), []):
                profile[t] += w
        scores = Counter()
        for t, w in profile.items():
            hit_items = inv.get(t, [])
            if len(hit_items) > 300:  # 过宽泛的标签降权，避免退化为热度
                share = w / np.log2(len(hit_items))
            else:
                share = w
            for i in hit_items:
                scores[i] += share
        tag_pred[u] = [i for i, _ in scores.most_common(args.topk)]

    rows = []
    for name, cold_channel in [("内容语义通道", content_pred), ("标签倒排通道", tag_pred)]:
        for m in args.quotas:
            fused = {}
            for u in users:
                v = content_pred[u] if name == "内容语义通道" else tag_pred[u]
                slots = list(v[:m])
                warm = [i for i in base_pred[u] if i not in cold_set]
                slots += [i for i in warm if i not in slots][: args.topk - len(slots)]
                fused[u] = slots
            met = evaluate_topk(u2gt, fused, ks=(20, 50))
            exposed = len({i for lst in fused.values() for i in lst if i in cold_set})
            rows.append({
                "冷启通道": name,
                "扶持额度(条)": m,
                "Recall@20": round(met["recall@20"], 4),
                "Recall@50": round(met["recall@50"], 4),
                "HitRate@50": round(met["hit@50"], 4),
                "NDCG@50": round(met["ndcg@50"], 4),
                "新内容曝光数": exposed,
                "新内容曝光率": round(exposed / len(cold_items), 4),
            })

    base_met = evaluate_topk(u2gt, base_pred, ks=(20, 50))
    rows.append({
        "冷启通道": "纯双塔（无冷启通道）",
        "扶持额度(条)": 0,
        "Recall@20": round(base_met["recall@20"], 4),
        "Recall@50": round(base_met["recall@50"], 4),
        "HitRate@50": round(base_met["hit@50"], 4),
        "NDCG@50": round(base_met["ndcg@50"], 4),
        "新内容曝光数": len({i for lst in base_pred.values() for i in lst if i in cold_set}),
        "新内容曝光率": round(len({i for lst in base_pred.values() for i in lst
                                  if i in cold_set}) / len(cold_items), 4),
    })

    save_json({"table": rows, "quota": args.quotas, "num_users": len(users),
               "num_cold_items": int(len(cold_items))}, DIR_RES / "coldstart_stage.json")
    print("\n=== 冷启动评测（%d 用户 / %d 件新内容）===" % (len(users), len(cold_items)))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
