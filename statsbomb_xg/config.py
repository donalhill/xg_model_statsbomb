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
# Removed redundant features:
#   - is_foot (r=-0.99 with is_header)
#   - gk_distance_to_shot (r=0.96 with distance_to_goal)
#   - is_open_play (r=-0.89 with is_set_piece; open play is implicit baseline)
FEATURE_COLUMNS = [
    # Shot geometry
    'distance_to_goal',
    'angle_to_goal',
    # Body part (is_header=0 implies foot)
    'is_header',
    # Shot context (open play is implicit when all are 0)
    'is_counter',
    'is_set_piece',
    'is_penalty',
    'is_first_time',
    'under_pressure',
    # Goalkeeper position (continuous)
    'gk_distance_from_goal_line',
    'gk_distance_from_center',
    'gk_positioning_error',
    # Defender position (continuous)
    'dist_nearest_defender',
    'dist_nearest_blocker',
    'goal_visible_pct',
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
