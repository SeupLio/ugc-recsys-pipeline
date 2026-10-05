"""分层实验（Layered Experiment）—— 哈希分桶与正交分层。

线上策略迭代的基础设施：把用户按 (layer, salt, user_id) 哈希进固定桶，
同一用户在同一层永远落同一桶（分流稳定性），不同层用不同 salt 保证
正交（一层的切分不系统性影响另一层——A 在 layer1 的分组与其在
layer2 的分组统计无关）。

- Experiments: 定义实验层与 arm（对照/实验组流量占比）
- assignment(user, layer) → arm 名称
- 流量守恒：各 arm 桶数之和 = 总桶数（hash space 均匀性的经验前提）
- 哈希用 sha1（稳定、无需第三方库；生产可用 xxhash/murmur 提速）

典型用法：
    exp = Experiments({
        "rank_model": {"salt": "L1", "buckets": 100,
                       "arms": {"control": 50, "din": 25, "esmm": 25}},
    })
    arm = exp.assignment(user_id=7, layer="rank_model")   # 永远同值
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LayerConfig:
    salt: str
    buckets: int = 100
    arms: Dict[str, int] = field(default_factory=dict)   # arm -> 桶数

    def __post_init__(self) -> None:
        total = sum(self.arms.values())
        if total != self.buckets:
            raise ValueError(
                f"层 arms 桶数之和 {total} != buckets {self.buckets}（流量必须守恒）")
        if any(n <= 0 for n in self.arms.values()):
            raise ValueError("arm 桶数必须为正")


class Experiments:
    def __init__(self, layers: Dict[str, LayerConfig | dict]) -> None:
        self._layers: Dict[str, LayerConfig] = {}
        for name, cfg in layers.items():
            cfg = LayerConfig(**cfg) if isinstance(cfg, dict) else cfg
            self._layers[name] = cfg

    def bucket(self, user_id: int, layer: str) -> int:
        """稳定分桶：sha1(salt:layer:uid) 取模。"""
        if layer not in self._layers:
            raise KeyError(f"未定义实验层: {layer}")
        cfg = self._layers[layer]
        key = f"{cfg.salt}:{layer}:{int(user_id)}".encode()
        return int(hashlib.sha1(key).hexdigest(), 16) % cfg.buckets

    def assignment(self, user_id: int, layer: str) -> str:
        """返回 arm 名称。落桶 → 按 arm 累积区间映射。"""
        b = self.bucket(user_id, layer)
        acc = 0
        for arm, n in self._layers[layer].arms.items():
            acc += n
            if b < acc:
                return arm
        return next(iter(self._layers[layer].arms))  # 理论不可达

    def layer_report(self, layer: str, user_ids: List[int]) -> Dict[str, float]:
        """给定用户集合，输出各 arm 流量占比（用于 SRM 自检）。"""
        n = max(1, len(user_ids))
        counts: Dict[str, int] = {a: 0 for a in self._layers[layer].arms}
        for uid in user_ids:
            counts[self.assignment(uid, layer)] += 1
        return {a: round(c / n, 4) for a, c in counts.items()}

    def layers(self) -> List[str]:
        return list(self._layers)
