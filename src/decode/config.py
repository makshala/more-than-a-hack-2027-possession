"""
Configuration constants and mappings for possession decoding.
"""
from typing import Dict

STATE_MAP: Dict[int, str] = {
    0: "H",
    1: "A",
    2: "D"
}

INV_STATE_MAP: Dict[str, int] = {
    "H": 0,
    "A": 1,
    "D": 2
}

DEFAULT_FPS: float = 25.0
EPS: float = 1e-12
