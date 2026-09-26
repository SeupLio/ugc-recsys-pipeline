#!/usr/bin/env python
"""Step 10：把各阶段的实验结果画成图（英文标签，避免不同环境缺中文字体导致豆腐块）。

用法：
    python scripts/10_make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def _load(name: str):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_recall() -> None:
    d = _load("recall_stage.json")
    if not d:
        return
    rows = d["table"]
    names = [r["召回通道"] for r in rows]
    en = ["Hot(30d decay)", "ItemCF(IUF)", "Two-Tower(Flat)", "RRF Fusion", "Two-Tower(IVF)"]
    r10 = [r["Recall@10"] for r in rows]
    r20 = [r["Recall@20"] for r in rows]
    r50 = [r["Recall@50"] for r in rows]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w, r10, w, label="Recall@10", color="#4C72B0")
    ax.bar(x, r20, w, label="Recall@20", color="#DD8452")
    ax.bar(x + w, r50, w, label="Recall@50", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(en, fontsize=9)
    ax.set_ylabel("Recall")
    ax.set_title("Recall Stage: multi-way candidate sources (test users=%d)" % d["num_test_users"])
    ax.legend()
    for i, v in enumerate(r50):
        ax.text(i + w, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "recall_stage.png")
    plt.close(fig)


def fig_rank() -> None:
    d = _load("rank_stage.json")
    if not d:
        return
    order = ["deepfm", "din", "esmm"]
    labels = ["DeepFM", "DIN", "ESMM"]
    aucs = [d[k]["metrics"]["ctr_auc"] for k in order]
    gaucs = [d[k]["metrics"]["ctr_gauc"] for k in order]
    ctcvr = [d[k]["metrics"].get("ctcvr_auc", np.nan) for k in order]
    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, aucs, w, label="CTR-AUC", color="#4C72B0")
    ax.bar(x, gaucs, w, label="CTR-GAUC", color="#DD8452")
    bars = ax.bar(x + w, np.nan_to_num(ctcvr), w, label="CTCVR-AUC", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("AUC")
    ax.set_title("Ranking stage: DeepFM vs DIN vs ESMM")
    for i, v in enumerate(np.nan_to_num(ctcvr)):
        if v > 0:
            ax.text(i + w, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.5)
    for i, v in enumerate(aucs):
        ax.text(i - w, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "rank_stage.png")
    plt.close(fig)


def fig_funnel() -> None:
    d = _load("funnel_stage.json")
    if not d:
        return
    rows = d["table"]
    labels = [r["阶段"] for r in rows]
    en = ["+DeepFM", "+DIN", "+ESMM", "DIN top10", "DIN+MMR(0.9)", "DIN+MMR(0.7)",
          "DIN+MMR(0.5)", "DIN+CategoryCap"]
    ndcg = [r["NDCG@10"] for r in rows]
    rec = [r["Recall@10"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(rows))
    ax1.plot(x, ndcg, "o-", color="#C44E52", label="NDCG@10")
    ax1.plot(x, rec, "s--", color="#4C72B0", label="Recall@10")
    ax1.axhline(d["recall_ceiling"], ls=":", color="gray",
                label=f"recall ceiling={d['recall_ceiling']:.3f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(en, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("metric")
    ax1.set_title("End-to-end funnel: recall -> rank -> rerank")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "funnel_stage.png")
    plt.close(fig)


def fig_coldstart() -> None:
    d = _load("coldstart_stage.json")
    if not d:
        return
    rows = [r for r in d["table"] if r["扶持额度(条)"] > 0 or "纯双塔" in r["冷启通道"]]
    content = [r for r in rows if r["冷启通道"] == "内容语义通道"]
    tag = [r for r in rows if r["冷启通道"] == "标签倒排通道"]
    base = [r for r in rows if "纯双塔" in r["冷启通道"]]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    if content:
        ax.plot([r["扶持额度(条)"] for r in content], [r["Recall@50"] for r in content],
                "o-", label="semantic content channel", color="#55A868")
    if tag:
        ax.plot([r["扶持额度(条)"] for r in tag], [r["Recall@50"] for r in tag],
                "s-", label="tag inverted-index channel", color="#DD8452")
    if base:
        ax.axhline(base[0]["Recall@50"], ls="--", color="#C44E52",
                   label=f"no cold-start channel ({base[0]['Recall@50']:.3f})")
    ax.set_xlabel("cold-start quota (slots per user)")
    ax.set_ylabel("Recall@50 on new content")
    ax.set_title("Cold start: how much recall does a dedicated new-content channel buy?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "coldstart.png")
    plt.close(fig)


def fig_growth() -> None:
    d = _load("growth_stage.json")
    if not d:
        return
    rows = d["lookalike"]
    en = ["centroid", "KNN vote", "random", "KNN+pLTV"]
    lifts = [r["相对大盘Lift"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    colors = ["#4C72B0", "#DD8452", "#999999", "#55A868"]
    bars = ax.bar(en, lifts, color=colors)
    ax.axhline(1.0, ls="--", color="gray")
    ax.set_ylabel("Lift vs random pool")
    ax.set_title("Lookalike: high-value user lift")
    for b, v, r in zip(bars, lifts, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}x ({r['人群规模']})",
                ha="center", fontsize=8)
    ax.set_ylim(0, max(lifts) * 1.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "growth.png")
    plt.close(fig)


def fig_ann() -> None:
    d = _load("ann_scaling.json")
    if not d:
        return
    sizes = sorted({r["库规模"] for r in d})
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for idx in ["Flat", "IVF1024_probe32", "HNSW32_ef64"]:
        vals = [r["单查询ms"] for r in d if r["索引"] == idx]
        if len(vals) == len(sizes):
            ax.plot(range(len(sizes)), vals, "o-", label=idx)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_yscale("log")
    ax.set_xlabel("corpus size (#items)")
    ax.set_ylabel("latency per query (ms, log scale)")
    ax.set_title("ANN scaling: when does brute force stop being enough?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "ann_scaling.png")
    plt.close(fig)


def fig_hard_negative() -> None:
    """难负例：两个评测口径必须一起看，否则会得出相反结论。"""
    d = _load("hard_negative.json")
    if not d:
        return
    res = d["results"]
    labels = [r["tag"] for r in res]
    easy = [r["auc_easy"] for r in res]
    hard = [r["auc_hard"] for r in res]
    x = np.arange(len(res))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, easy, w, label="easy-neg eval (popularity)", color="#999999")
    ax.bar(x + w / 2, hard, w, label="hard-neg eval (recall top-K)", color="#C44E52")
    for i, (e, h) in enumerate(zip(easy, hard)):
        ax.text(i - w / 2, e + 0.01, f"{e:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, h + 0.01, f"{h:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, ls=":", color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("AUC")
    ax.set_title("Hard-negative mining: baseline is near-random (0.52) on realistic negatives")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "hard_negative.png")
    plt.close(fig)


def fig_quality_tradeoff() -> None:
    """质量降权的精度↔覆盖权衡：α 越大精度略升但覆盖率下降。"""
    d = _load("intent_quality_stage.json")
    if not d:
        return
    rows = d["quality_rerank"]
    alphas = [r["质量权重α"] for r in rows]
    rec = [r["Recall@10"] for r in rows]
    cov = [r["覆盖率↑"] for r in rows]
    q = [r["Top10平均质量分"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.5, 4))
    ax1.plot(alphas, rec, "o-", color="#4C72B0", label="Recall@10")
    ax1.plot(alphas, q, "s-", color="#55A868", label="avg quality of Top10")
    ax1.set_xlabel("quality penalty weight α")
    ax1.set_ylabel("Recall@10 / quality")
    ax2 = ax1.twinx()
    ax2.plot(alphas, cov, "^--", color="#C44E52", label="catalog coverage")
    ax2.set_ylabel("coverage", color="#C44E52")
    ax1.set_title("Quality-aware rerank: precision vs catalog coverage")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(ASSETS / "quality_rerank.png")
    plt.close(fig)


def main() -> None:
    for fn in [fig_recall, fig_rank, fig_funnel, fig_coldstart, fig_growth, fig_ann,
               fig_hard_negative, fig_quality_tradeoff]:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"{fn.__name__} failed: {exc}")
    print("figures ->", ASSETS)
    for p in sorted(ASSETS.glob("*.png")):
        print(" ", p.name, f"{p.stat().st_size / 1024:.0f}KB")


if __name__ == "__main__":
    main()
