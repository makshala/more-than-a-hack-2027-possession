"""
Configuration and domain-grounded physical constants for Stage 1 Tracking ETL.
"""

import os
import sys

# ---------------------------------------------------------
# 1. REPOSITORY ROOT & GAMES.PY INTEGRATION
# ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if not os.path.exists(os.path.join(ROOT, "games.py")) and os.path.exists(os.path.join(os.getcwd(), "games.py")):
    ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import games
    GAME = getattr(games, "GAME", "BAR-ATH")
    PITCH_LENGTH, PITCH_WIDTH = games.GAMES[GAME]["pitch"]
    FPS = float(games.GAMES[GAME]["fps"])
except Exception:
    games = None
    GAME = "BAR-ATH"
    PITCH_LENGTH, PITCH_WIDTH = 105.3, 68.0
    FPS = 25.0

DT = 1.0 / FPS

# Domain-grounded physical constants
SHORT_GAP_MAX_FRAMES = 25
CONTESTED_RADIUS_M = 2.5
CONTROL_RADIUS_3M = 3.0
SUPPORT_RADIUS_5M = 5.0
CLONE_DIST_THRESHOLD_M = 0.45
WINDOW_2S_FRAMES = 50
WINDOW_4S_FRAMES = 100
REACTION_TIME_S = 0.40
PLAYER_MAX_SPEED_MS = 7.50
PITCH_CONTROL_TAU_S = 0.50
MAX_BALL_SPEED_MS = 38.0
SAVGOL_WINDOW = 7
SAVGOL_POLY = 2

# Official frame boundaries
H1_START, H1_END = 11185, 90651
H2_START, H2_END = 110808, 190197
KICKOFF_2ND_HALF = 113742

HOME, AWAY = 0, 1