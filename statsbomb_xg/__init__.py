"""
StatsBomb xG Model Package

A focused xG (Expected Goals) model using StatsBomb open data from La Liga 2015/16.
"""

from .data import load_shots
from .features import engineer_features
from .model import train_xg_model, predict_xg, load_model, save_model
from .evaluate import temporal_split, evaluate_model
from .player_analysis import get_player_shots, player_summary, griezmann_analysis

__all__ = [
    'load_shots',
    'engineer_features',
    'train_xg_model',
    'predict_xg',
    'load_model',
    'save_model',
    'temporal_split',
    'evaluate_model',
    'get_player_shots',
    'player_summary',
    'griezmann_analysis',
]
