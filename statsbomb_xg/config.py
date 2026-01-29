"""
Configuration constants for xG model.
"""
from pathlib import Path

# StatsBomb IDs
COMPETITION_ID = 11  # La Liga
SEASON_ID = 27       # 2015/16

# Temporal split
TRAIN_END_DATE = "2016-01-01"

# Feature columns for model
# Minimal set based on SHAP analysis - only features with meaningful impact
FEATURE_COLUMNS = [
    'distance_to_goal',       # Shot geometry - dominant feature
    'angle_to_goal',          # Shot geometry - second most important
    'is_header',              # Body part - clear negative effect on xG
    'gk_distance_from_goal_line',  # GK position - captures 1v1 situations
    'dist_nearest_defender',  # Defensive pressure
    'goal_visible_pct',       # Goal visibility accounting for blockers
]

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Pitch dimensions (StatsBomb uses 120x80 yards)
PITCH_LENGTH = 120
PITCH_WIDTH = 80
GOAL_CENTER_X = 120
GOAL_CENTER_Y = 40
GOAL_WIDTH = 8  # yards
