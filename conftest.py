"""pytest 全局配置：把 src 与 scripts 加入 sys.path。

有了它，`python -m pytest tests -q` 无需再手动设置 PYTHONPATH=src。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT / "scripts"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
