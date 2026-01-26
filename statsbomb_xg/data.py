"""
Data loading and caching for StatsBomb open data.
"""
import pandas as pd
import pickle
from pathlib import Path
from statsbombpy import sb

from .config import COMPETITION_ID, SEASON_ID, DATA_DIR


def load_shots(use_cache=True):
    """
    Load all shots from La Liga 2015/16.

    Caches to pickle for faster subsequent loads.

    Returns:
        pd.DataFrame: All shots with relevant columns
    """
    cache_path = DATA_DIR / "laliga_2015_16_shots.pkl"

    if use_cache and cache_path.exists():
        print("Loading shots from cache...")
        return pd.read_pickle(cache_path)

    print("Loading shots from StatsBomb API...")

    # Get all matches
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    print(f"Found {len(matches)} matches")

    all_shots = []

    for _, match in matches.iterrows():
        match_id = match['match_id']
        match_date = match['match_date']
        home_team = match['home_team']
        away_team = match['away_team']

        try:
            events = sb.events(match_id=match_id)
            shots = events[events['type'] == 'Shot'].copy()

            if len(shots) == 0:
                continue

            # Add match context
            shots['match_id'] = match_id
            shots['match_date'] = pd.to_datetime(match_date)
            shots['home_team'] = home_team
            shots['away_team'] = away_team

            all_shots.append(shots)

        except Exception as e:
            print(f"Error loading match {match_id}: {e}")
            continue

    df = pd.concat(all_shots, ignore_index=True)
    print(f"Loaded {len(df)} total shots")

    # Extract key columns
    df = _extract_shot_data(df)

    # Cache for future use
    df.to_pickle(cache_path)
    print(f"Cached to {cache_path}")

    return df


def _extract_shot_data(df):
    """Extract and flatten relevant shot data."""
    # Build new columns efficiently using assign
    result = pd.DataFrame({
        'id': df['id'],
        'match_id': df['match_id'],
        'match_date': df['match_date'],
        'minute': df['minute'],
        'second': df['second'],
        'player': df['player'],
        'team': df['team'],
        'home_team': df['home_team'],
        'away_team': df['away_team'],
        'x': df['location'].apply(lambda loc: loc[0] if isinstance(loc, list) else None),
        'y': df['location'].apply(lambda loc: loc[1] if isinstance(loc, list) else None),
        'is_goal': (df['shot_outcome'] == 'Goal').astype(int),
        'body_part': df['shot_body_part'].fillna('Unknown'),
        'shot_type': df['shot_type'].fillna('Open Play'),
        'statsbomb_xg': df['shot_statsbomb_xg'].fillna(0),
        'under_pressure': df['under_pressure'].fillna(False).astype(int),
        'shot_first_time': df['shot_first_time'].fillna(False),
        'freeze_frame': df['shot_freeze_frame'],
        'play_pattern': df.get('play_pattern', pd.Series(['Regular Play'] * len(df))),
    })

    return result


def load_matches():
    """Load match metadata."""
    return sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
