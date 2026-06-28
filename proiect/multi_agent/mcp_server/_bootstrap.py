"""
Bootstrap sys.path — pune `skillab-py/src` și `src/` pe path, ca în `main.py`.

Modulele MCP rulează „flat" (ca scripturi, nu ca pachet), deci import-ul
`import _bootstrap` la începutul fiecărui entry-point asigură că
`from analyst_agent import ...`, `from skillab import ...` etc. se rezolvă.
Idempotent — se poate importa de oricâte ori.
"""
import sys
from pathlib import Path

_MULTI_AGENT = Path(__file__).resolve().parent.parent  # .../multi_agent

for _p in (_MULTI_AGENT / "skillab-py" / "src", _MULTI_AGENT / "src"):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
