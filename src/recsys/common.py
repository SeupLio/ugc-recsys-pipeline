"""公共工具包：路径常量、随机种子、配置加载、计时与日志。"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# ---------------------------------------------------------------- 路径常量
ROOT = Path(__file__).resolve().parents[2]
DIR_RAW = ROOT / "data" / "raw"
DIR_PROC = ROOT / "data" / "processed"
DIR_RES = ROOT / "results"
DIR_CKPT = ROOT / "checkpoints"
DIR_LOG = ROOT / "logs"

for _d in (DIR_PROC, DIR_RES, DIR_CKPT, DIR_LOG):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 随机种子
def seed_everything(seed: int = 42) -> None:
    """固定 Python / NumPy / PyTorch 的随机源，保证实验可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# ---------------------------------------------------------------- 配置
def load_config(path: str | Path) -> Dict[str, Any]:
    """读取 YAML 配置；无 PyYAML 时自动退回同名 JSON。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore

            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            json_path = path.with_suffix(".json")
            if not json_path.exists():
                raise RuntimeError("未安装 PyYAML，且找不到同名 JSON 配置")
            with json_path.open("r", encoding="utf-8") as f:
                return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------- 日志 / 计时
_FMT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"


def get_logger(name: str = "recsys", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    logger.addHandler(sh)
    fh = logging.FileHandler(DIR_LOG / f"{name}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


@contextmanager
def timer(desc: str = "", logger: logging.Logger | None = None):
    """统计一段代码的墙钟耗时。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        cost = time.perf_counter() - t0
        msg = f"{desc} 耗时 {cost:.3f}s" if desc else f"耗时 {cost:.3f}s"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
