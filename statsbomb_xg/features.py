"""
Feature engineering for xG model.

Industry-standard features based on StatsBomb methodology:
- Continuous measures instead of discrete counts (avoids binary artifacts)
- Goalkeeper positioning features
- Proportion of goal visible (not just blocker count)
- Distance to nearest defender (not just count within radius)
"""
import numpy as np
import pandas as pd

from .config import GOAL_CENTER_X, GOAL_CENTER_Y, GOAL_WIDTH


# Goal post positions (constant)
GOAL_Y_LEFT = GOAL_CENTER_Y - GOAL_WIDTH / 2   # 36
GOAL_Y_RIGHT = GOAL_CENTER_Y + GOAL_WIDTH / 2  # 44


def distance_to_goal(x, y):
    """Distance from shot location to goal center."""
    return np.sqrt((GOAL_CENTER_X - x)**2 + (GOAL_CENTER_Y - y)**2)


def angle_to_goal(x, y):
    """
    Angle subtended by goal posts from shot location (degrees).
    Larger angle = more of goal visible.
    """
    dx = GOAL_CENTER_X - x
    dy_left = GOAL_Y_LEFT - y
    dy_right = GOAL_Y_RIGHT - y

    angle_left = np.arctan2(dy_left, dx)
    angle_right = np.arctan2(dy_right, dx)

    return np.degrees(np.abs(angle_right - angle_left))


# =============================================================================
# Goalkeeper Features (continuous)
# =============================================================================

def _get_goalkeeper(freeze_frame):
    """Extract goalkeeper from freeze frame (opponent GK)."""
    if not isinstance(freeze_frame, list):
        return None

    for player in freeze_frame:
        # Opponent goalkeeper
        if not player.get('teammate', True):
            position = player.get('position', {})
            if position.get('name') == 'Goalkeeper' or position.get('id') == 1:
                loc = player.get('location')
                if loc and len(loc) >= 2:
                    return {'x': loc[0], 'y': loc[1]}
    return None


def gk_distance_from_goal_line(freeze_frame):
    """
    How far the goalkeeper is from the goal line (off their line).

    Returns distance in yards. Higher = GK further out = smaller target.
    Returns NaN if no GK found.
    """
    gk = _get_goalkeeper(freeze_frame)
    if gk is None:
        return np.nan
    return GOAL_CENTER_X - gk['x']


def gk_distance_from_goal_center(freeze_frame):
    """
    How far the goalkeeper is from the center of the goal (y-axis).

    Returns absolute distance. Higher = GK off-center = more goal exposed.
    Returns NaN if no GK found.
    """
    gk = _get_goalkeeper(freeze_frame)
    if gk is None:
        return np.nan
    return abs(gk['y'] - GOAL_CENTER_Y)


def gk_distance_to_shot(x, y, freeze_frame):
    """
    Distance from shooter to goalkeeper.

    Returns NaN if no GK found.
    """
    gk = _get_goalkeeper(freeze_frame)
    if gk is None:
        return np.nan
    return np.sqrt((gk['x'] - x)**2 + (gk['y'] - y)**2)


def gk_positioning_error(x, y, freeze_frame):
    """
    How far the GK is from the optimal position on the shot line.

    Optimal position is on the line between shot and goal center,
    at the GK's current x-position. This measures lateral error.

    Returns distance in yards. Higher = worse positioning.
    Returns NaN if no GK found.
    """
    gk = _get_goalkeeper(freeze_frame)
    if gk is None:
        return np.nan

    # If GK is at or behind goal line, can't calculate
    if gk['x'] >= GOAL_CENTER_X:
        return 0.0

    # Line from shot to goal center
    # At GK's x position, where should they be on y-axis?
    t = (gk['x'] - x) / (GOAL_CENTER_X - x) if (GOAL_CENTER_X - x) != 0 else 0
    optimal_y = y + t * (GOAL_CENTER_Y - y)

    return abs(gk['y'] - optimal_y)


# =============================================================================
# Defender Features (continuous)
# =============================================================================

def _get_opponents(freeze_frame, exclude_gk=True):
    """Get list of opponent positions from freeze frame."""
    if not isinstance(freeze_frame, list):
        return []

    opponents = []
    for player in freeze_frame:
        if not player.get('teammate', True):
            position = player.get('position', {})
            is_gk = position.get('name') == 'Goalkeeper' or position.get('id') == 1

            if exclude_gk and is_gk:
                continue

            loc = player.get('location')
            if loc and len(loc) >= 2:
                opponents.append({'x': loc[0], 'y': loc[1]})

    return opponents


def distance_to_nearest_defender(x, y, freeze_frame):
    """
    Distance from shooter to nearest outfield defender.

    Continuous measure of pressure on the shooter.
    Returns NaN if no defenders found.
    """
    opponents = _get_opponents(freeze_frame, exclude_gk=True)
    if not opponents:
        return np.nan

    distances = [np.sqrt((opp['x'] - x)**2 + (opp['y'] - y)**2) for opp in opponents]
    return min(distances)


def _point_in_triangle(px, py, x1, y1, x2, y2, x3, y3):
    """Check if point (px, py) is inside triangle defined by three vertices."""
    def sign(p1x, p1y, p2x, p2y, p3x, p3y):
        return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)

    d1 = sign(px, py, x1, y1, x2, y2)
    d2 = sign(px, py, x2, y2, x3, y3)
    d3 = sign(px, py, x3, y3, x1, y1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def goal_visible_proportion(x, y, freeze_frame, n_samples=100):
    """
    Proportion of the goal that is not blocked by defenders or GK.

    Continuous measure (0-1). Uses ray-casting across the goal width.
    1.0 = entire goal visible, 0.0 = completely blocked.

    This is more nuanced than counting blockers.
    """
    if not isinstance(freeze_frame, list) or len(freeze_frame) == 0:
        return 1.0  # No freeze frame = assume open

    # Get all opponents including GK
    blockers = []
    for player in freeze_frame:
        if not player.get('teammate', True):
            loc = player.get('location')
            if loc and len(loc) >= 2:
                blockers.append({'x': loc[0], 'y': loc[1]})

    if not blockers:
        return 1.0

    # Sample points across the goal
    goal_points = np.linspace(GOAL_Y_LEFT, GOAL_Y_RIGHT, n_samples)
    visible_count = 0

    # Player blocking radius (approximate body width)
    BLOCK_RADIUS = 0.5  # yards

    for goal_y in goal_points:
        # Check if line from shot to this goal point is blocked
        blocked = False

        for blocker in blockers:
            bx, by = blocker['x'], blocker['y']

            # Only consider blockers between shooter and goal
            if bx <= x or bx >= GOAL_CENTER_X:
                continue

            # Calculate perpendicular distance from blocker to shot line
            # Line from (x, y) to (GOAL_CENTER_X, goal_y)
            line_dx = GOAL_CENTER_X - x
            line_dy = goal_y - y
            line_len = np.sqrt(line_dx**2 + line_dy**2)

            if line_len == 0:
                continue

            # Vector from shot to blocker
            to_blocker_x = bx - x
            to_blocker_y = by - y

            # Project blocker onto line
            t = (to_blocker_x * line_dx + to_blocker_y * line_dy) / (line_len**2)

            # Only if blocker projection is between shot and goal
            if 0 < t < 1:
                # Closest point on line to blocker
                closest_x = x + t * line_dx
                closest_y = y + t * line_dy

                # Distance from blocker to line
                dist_to_line = np.sqrt((bx - closest_x)**2 + (by - closest_y)**2)

                if dist_to_line < BLOCK_RADIUS:
                    blocked = True
                    break

        if not blocked:
            visible_count += 1

    return visible_count / n_samples


def distance_to_nearest_blocker(x, y, freeze_frame):
    """
    Distance to nearest defender/GK in the shooting cone.

    Only considers players between shooter and goal in the cone.
    Continuous alternative to "defenders in cone" count.
    Returns NaN if no blockers in cone.
    """
    if not isinstance(freeze_frame, list) or len(freeze_frame) == 0:
        return np.nan

    min_dist = np.inf

    for player in freeze_frame:
        if not player.get('teammate', True):  # Opponent
            loc = player.get('location')
            if not loc or len(loc) < 2:
                continue

            px, py = loc[0], loc[1]

            # Must be between shooter and goal
            if px <= x or px >= GOAL_CENTER_X:
                continue

            # Check if in the cone (triangle from shot to goalposts)
            t = (px - x) / (GOAL_CENTER_X - x)
            y_min = y + t * (GOAL_Y_LEFT - y)
            y_max = y + t * (GOAL_Y_RIGHT - y)

            if min(y_min, y_max) <= py <= max(y_min, y_max):
                dist = np.sqrt((px - x)**2 + (py - y)**2)
                min_dist = min(min_dist, dist)

    return min_dist if min_dist != np.inf else np.nan


# =============================================================================
# Main Feature Engineering Function
# =============================================================================

def engineer_features(df):
    """
    Add all engineered features to the dataframe.

    Features follow industry best practices:
    - Continuous measures (not discrete counts)
    - Goalkeeper positioning
    - Proportion-based blocking measure
    """
    df = df.copy()

    # ----- Shot geometry (always available) -----
    df['distance_to_goal'] = df.apply(
        lambda row: distance_to_goal(row['x'], row['y']), axis=1
    )
    df['angle_to_goal'] = df.apply(
        lambda row: angle_to_goal(row['x'], row['y']), axis=1
    )

    # ----- Body part -----
    df['is_header'] = (df['body_part'] == 'Head').astype(int)
    df['is_foot'] = df['body_part'].isin(['Left Foot', 'Right Foot']).astype(int)

    # ----- Shot context -----
    df['is_open_play'] = (df['shot_type'] == 'Open Play').astype(int)
    df['is_penalty'] = (df['shot_type'] == 'Penalty').astype(int)
    df['is_set_piece'] = df['shot_type'].isin(
        ['Free Kick', 'Corner', 'Kick Off']
    ).astype(int)

    play_pattern = df.get('play_pattern', pd.Series(['Regular Play'] * len(df)))
    df['is_counter'] = (play_pattern == 'From Counter').astype(int)

    df['is_first_time'] = df['shot_first_time'].fillna(False).astype(int)
    df['under_pressure'] = df['under_pressure'].fillna(0).astype(int)

    # ----- Goalkeeper features (continuous) -----
    df['gk_distance_from_goal_line'] = df['freeze_frame'].apply(
        gk_distance_from_goal_line
    )
    df['gk_distance_from_center'] = df['freeze_frame'].apply(
        gk_distance_from_goal_center
    )
    df['gk_distance_to_shot'] = df.apply(
        lambda row: gk_distance_to_shot(row['x'], row['y'], row['freeze_frame']),
        axis=1
    )
    df['gk_positioning_error'] = df.apply(
        lambda row: gk_positioning_error(row['x'], row['y'], row['freeze_frame']),
        axis=1
    )

    # ----- Defender features (continuous) -----
    df['dist_nearest_defender'] = df.apply(
        lambda row: distance_to_nearest_defender(row['x'], row['y'], row['freeze_frame']),
        axis=1
    )
    df['dist_nearest_blocker'] = df.apply(
        lambda row: distance_to_nearest_blocker(row['x'], row['y'], row['freeze_frame']),
        axis=1
    )
    df['goal_visible_pct'] = df.apply(
        lambda row: goal_visible_proportion(row['x'], row['y'], row['freeze_frame']),
        axis=1
    )

    # ----- Handle NaN values -----
    # Fill NaN with reasonable defaults for missing freeze frame data
    # GK features: assume GK on goal line, centered (conservative)
    df['gk_distance_from_goal_line'] = df['gk_distance_from_goal_line'].fillna(0)
    df['gk_distance_from_center'] = df['gk_distance_from_center'].fillna(0)
    df['gk_distance_to_shot'] = df['gk_distance_to_shot'].fillna(
        df['distance_to_goal']  # Approximate with shot distance
    )
    df['gk_positioning_error'] = df['gk_positioning_error'].fillna(0)

    # Defender features: assume no immediate pressure if missing
    df['dist_nearest_defender'] = df['dist_nearest_defender'].fillna(10.0)
    df['dist_nearest_blocker'] = df['dist_nearest_blocker'].fillna(
        df['distance_to_goal']  # No blocker = can shoot freely
    )
    df['goal_visible_pct'] = df['goal_visible_pct'].fillna(1.0)

    return df
