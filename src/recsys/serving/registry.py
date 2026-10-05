"""模型注册表（Model Registry）—— 版本元数据 / stage 迁移 / 原子切换与回滚。

线上换模型不能「kill 掉进程改个路径」：需要可追溯（谁在何时把哪版推上
生产、指标如何）、可回滚（30 秒内切回上一版）、可审计。本模块：

- ModelVersion：版本元数据（权重路径、指标、训练数据快照引用）
- stage 流转：none → staging（影子验证）→ production → archived
- activate(name) 原子切换 + lock 保护 + 自动记录历史
- rollback() 一键回上一版（保留完整切换历史）
- 影子模式：staging 模型可并行打分不上线（shadow=True 时只记指标）

注册表落盘 registry.json（进程崩溃后重建状态），写入带原子性：
先写临时文件再 os.replace，避免半截 JSON。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class RegistryError(Exception):
    pass


VALID_STAGES = ("none", "staging", "production", "archived")


class ModelRegistry:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._lock = threading.RLock()
        self._path = Path(path) if path else None
        self._models: Dict[str, Dict[str, Any]] = {}
        self._history: List[Dict[str, Any]] = []
        if self._path and self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._models = data.get("models", {})
                self._history = data.get("history", [])
            except (json.JSONDecodeError, OSError):
                pass

    # ---------------- 注册 ----------------

    def register(self, name: str, weights: str, metrics: Optional[dict] = None,
                 stage: str = "none", shadow: bool = False) -> Dict[str, Any]:
        if stage not in VALID_STAGES:
            raise RegistryError(f"非法 stage: {stage}")
        with self._lock:
            if name in self._models:
                raise RegistryError(f"模型已注册: {name}")
            self._models[name] = {
                "name": name, "weights": weights,
                "metrics": metrics or {}, "stage": stage,
                "shadow": bool(shadow), "created_at": time.time(),
                "activated_at": None,
            }
            self._persist()
            return dict(self._models[name])

    # ---------------- 查询 ----------------

    def info(self, name: str) -> Dict[str, Any]:
        with self._lock:
            if name not in self._models:
                raise RegistryError(f"未注册模型: {name}")
            return dict(self._models[name])

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(m) for _, m in sorted(self._models.items())]

    def production(self) -> Optional[str]:
        with self._lock:
            for m in self._models.values():
                if m["stage"] == "production" and not m["shadow"]:
                    return m["name"]
            return None

    # ---------------- 流转 ----------------

    def _transition(self, name: str, stage: str, note: str = "") -> Dict[str, Any]:
        with self._lock:
            if name not in self._models:
                raise RegistryError(f"未注册模型: {name}")
            m = self._models[name]
            old = m["stage"]
            m["stage"] = stage
            if stage == "production":
                m["activated_at"] = time.time()
            self._history.append({"ts": time.time(), "model": name,
                                  "from": old, "to": stage, "note": note})
            self._persist()
            return dict(m)

    def activate(self, name: str, note: str = "") -> Dict[str, Any]:
        """原子上线：旧 production → archived，新模型 → production。

        流转护栏：stage 仍为 none（从未做过 shadow 验证）的模型拒绝上线；
        从 staging / archived 上线均合法（后者对应回滚路径）。
        """
        with self._lock:
            if name not in self._models:
                raise RegistryError(f"未注册模型: {name}")
            if self._models[name]["stage"] == "none":
                raise RegistryError(
                    f"{name} 未经过 staging 影子验证，拒绝直接上线")
            cur = self.production()
            if cur == name:
                raise RegistryError(f"{name} 已是 production")
            if cur:
                self._transition(cur, "archived", note=f"被 {name} 替换")
            # 上线即转为正式服务（shadow 标记清除，影子期结束）
            self._models[name]["shadow"] = False
            return self._transition(name, "production", note=note)

    def to_staging(self, name: str, shadow: bool = True) -> Dict[str, Any]:
        with self._lock:
            if name not in self._models:
                raise RegistryError(f"未注册模型: {name}")
            self._models[name]["shadow"] = bool(shadow)
            return self._transition(name, "staging", note="shadow 验证")

    def rollback(self) -> Optional[str]:
        """一键回滚到最近一次被替换下线的模型。"""
        with self._lock:
            for h in reversed(self._history):
                if h["to"] == "archived":
                    cand = h["model"]
                    if cand in self._models:
                        self.activate(cand, note="自动回滚")
                        return cand
            return None

    def record_metrics(self, name: str, metrics: Dict[str, float]) -> None:
        with self._lock:
            if name not in self._models:
                raise RegistryError(f"未注册模型: {name}")
            self._models[name]["metrics"].update(metrics)
            self._persist()

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    # ---------------- 内部 ----------------

    def _persist(self) -> None:
        if not self._path:
            return
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"models": self._models,
                                   "history": self._history},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)
