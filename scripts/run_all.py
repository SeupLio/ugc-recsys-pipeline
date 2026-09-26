#!/usr/bin/env python
"""一键复现：数据 → 内容向量 → 召回 → 精排 → 漏斗 → 冷启动 → 增长 → ANN 规模效应 → 单测。

用法：
    python scripts/run_all.py            # 全流程
    python scripts/run_all.py --epochs_tt 8 --epochs_rank 4 --skip_tests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("数据准备", ["scripts/01_prepare_data.py"]),
    ("内容理解(语义向量+标签)", ["scripts/02_build_content.py", "--device", "auto"]),
    ("双塔召回训练", ["scripts/03_train_two_tower.py"]),
    ("召回层评测", ["scripts/04_eval_recall.py"]),
    ("ANN 规模效应", ["scripts/04b_ann_scaling.py"]),
    ("精排/多目标训练", ["scripts/05_train_rank.py", "--model", "all"]),
    ("全链路漏斗评测", ["scripts/06_eval_funnel.py"]),
    ("冷启动评测", ["scripts/07_eval_coldstart.py"]),
    ("用户增长评测", ["scripts/08_eval_growth.py"]),
    ("单元测试", ["-m", "pytest", "tests", "-q"]),
]


def run(name: str, args: list[str], timeout: int | None = None) -> bool:
    print(f"\n{'=' * 72}\n▶ {name}\n{'=' * 72}")
    t0 = time.perf_counter()
    proc = subprocess.run([PY] + args, cwd=ROOT, timeout=timeout)
    cost = time.perf_counter() - t0
    ok = proc.returncode == 0
    print(f"◀ {name} {'成功' if ok else '失败'}，耗时 {cost:.1f}s")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs_tt", type=int, default=8)
    p.add_argument("--epochs_rank", type=int, default=4)
    p.add_argument("--skip_tests", action="store_true")
    a = p.parse_args()

    failed = []
    for name, cmd in STEPS:
        if "03_train_two_tower" in cmd[0]:
            cmd = cmd + ["--epochs", str(a.epochs_tt)]
        if "05_train_rank" in cmd[0]:
            cmd = cmd + ["--epochs", str(a.epochs_rank)]
        if a.skip_tests and cmd[:2] == ["-m", "pytest"]:
            continue
        if not run(name, cmd):
            failed.append(name)
    print("\n" + "=" * 72)
    print("全部阶段完成 ✅" if not failed else f"失败阶段: {failed} ❌")


if __name__ == "__main__":
    main()
