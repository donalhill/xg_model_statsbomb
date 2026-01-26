"""
Player-level xG analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from .config import OUTPUT_DIR


def get_player_shots(df, player_name):
    """
    Get all shots for a specific player.

    Args:
        df: DataFrame with all shots
        player_name: Player name to filter (partial match)

    Returns:
        DataFrame of player's shots
    """
    mask = df['player'].str.contains(player_name, case=False, na=False)
    player_df = df[mask].copy()

    if len(player_df) == 0:
        print(f"No shots found for player: {player_name}")
        return pd.DataFrame()

    print(f"Found {len(player_df)} shots for {player_df['player'].iloc[0]}")
    return player_df


def player_summary(df_player, model=None, feature_columns=None):
    """
    Generate summary statistics for a player.

    Args:
        df_player: DataFrame of player's shots
        model: Trained xG model (optional, uses statsbomb_xg if not provided)
        feature_columns: Feature columns for model prediction

    Returns:
        dict of summary statistics
    """
    if len(df_player) == 0:
        return {}

    # Basic stats
    total_shots = len(df_player)
    total_goals = df_player['is_goal'].sum()
    conversion_rate = total_goals / total_shots if total_shots > 0 else 0

    # xG
    if model is not None and feature_columns is not None:
        X = df_player[feature_columns]
        our_xg = model.predict_proba(X)[:, 1]
        total_our_xg = our_xg.sum()
    else:
        our_xg = None
        total_our_xg = None

    sb_xg = df_player['statsbomb_xg'].sum()

    summary = {
        'player': df_player['player'].iloc[0],
        'total_shots': total_shots,
        'total_goals': int(total_goals),
        'conversion_rate': conversion_rate,
        'statsbomb_xg': sb_xg,
        'goals_minus_sb_xg': total_goals - sb_xg,
    }

    if total_our_xg is not None:
        summary['our_xg'] = total_our_xg
        summary['goals_minus_our_xg'] = total_goals - total_our_xg

    return summary


def print_player_summary(summary):
    """Pretty print player summary."""
    if not summary:
        return

    print(f"\n{'='*50}")
    print(f"Player: {summary['player']}")
    print(f"{'='*50}")
    print(f"Total Shots:       {summary['total_shots']}")
    print(f"Total Goals:       {summary['total_goals']}")
    print(f"Conversion Rate:   {summary['conversion_rate']:.1%}")
    print(f"StatsBomb xG:      {summary['statsbomb_xg']:.2f}")
    print(f"Goals - SB xG:     {summary['goals_minus_sb_xg']:+.2f}")

    if 'our_xg' in summary:
        print(f"Our Model xG:      {summary['our_xg']:.2f}")
        print(f"Goals - Our xG:    {summary['goals_minus_our_xg']:+.2f}")


def plot_shot_map(df_player, model=None, feature_columns=None, save=True, filename=None):
    """
    Plot shot map for a player on a football pitch.

    Goals are shown as stars, misses as circles.
    Color intensity shows xG value.
    """
    if len(df_player) == 0:
        return None

    player_name = df_player['player'].iloc[0]

    # Get xG values
    if model is not None and feature_columns is not None:
        X = df_player[feature_columns]
        xg_values = model.predict_proba(X)[:, 1]
    else:
        xg_values = df_player['statsbomb_xg'].values

    # Create pitch
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass',
                  line_color='white', stripe=True)
    fig, ax = pitch.draw(figsize=(12, 8))

    # Separate goals and non-goals
    goals = df_player['is_goal'] == 1
    non_goals = df_player['is_goal'] == 0

    # Plot non-goals (circles)
    if non_goals.sum() > 0:
        scatter_miss = ax.scatter(
            df_player.loc[non_goals, 'x'],
            df_player.loc[non_goals, 'y'],
            c=xg_values[non_goals],
            cmap='Reds',
            s=100,
            edgecolors='black',
            linewidth=1,
            alpha=0.7,
            vmin=0, vmax=0.5,
            marker='o',
            label='Miss/Saved'
        )

    # Plot goals (stars)
    if goals.sum() > 0:
        scatter_goal = ax.scatter(
            df_player.loc[goals, 'x'],
            df_player.loc[goals, 'y'],
            c=xg_values[goals],
            cmap='Reds',
            s=200,
            edgecolors='gold',
            linewidth=2,
            alpha=1.0,
            vmin=0, vmax=0.5,
            marker='*',
            label='Goal'
        )

    # Colorbar
    if non_goals.sum() > 0:
        plt.colorbar(scatter_miss, ax=ax, label='xG', shrink=0.6)
    elif goals.sum() > 0:
        plt.colorbar(scatter_goal, ax=ax, label='xG', shrink=0.6)

    # Title and legend
    total_goals = goals.sum()
    total_xg = xg_values.sum()
    ax.set_title(f"{player_name}\n{len(df_player)} shots, {total_goals} goals, {total_xg:.1f} xG")
    ax.legend(loc='upper left')

    if save:
        if filename is None:
            safe_name = player_name.replace(' ', '_').lower()
            filename = f"shot_map_{safe_name}.png"
        path = OUTPUT_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved shot map to {path}")

    return fig


def plot_cumulative_xg(df_player, model=None, feature_columns=None, save=True, filename=None):
    """
    Plot cumulative goals vs cumulative xG over the season.

    Shows if player is over/under-performing their xG.
    """
    if len(df_player) == 0:
        return None

    player_name = df_player['player'].iloc[0]

    # Sort by date
    df_sorted = df_player.sort_values('match_date').copy()

    # Get xG values
    if model is not None and feature_columns is not None:
        X = df_sorted[feature_columns]
        xg_values = model.predict_proba(X)[:, 1]
    else:
        xg_values = df_sorted['statsbomb_xg'].values

    # Calculate cumulative sums
    cumulative_goals = df_sorted['is_goal'].cumsum()
    cumulative_xg = np.cumsum(xg_values)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    shots = range(1, len(df_sorted) + 1)
    ax.plot(shots, cumulative_goals, 'b-', linewidth=2, label='Actual Goals')
    ax.plot(shots, cumulative_xg, 'r--', linewidth=2, label='Expected Goals (xG)')
    ax.fill_between(shots, cumulative_goals, cumulative_xg,
                    alpha=0.3, color='green' if cumulative_goals.iloc[-1] > cumulative_xg[-1] else 'red')

    ax.set_xlabel('Shot Number')
    ax.set_ylabel('Cumulative Goals / xG')
    ax.set_title(f'{player_name} - Cumulative Goals vs xG')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add final values annotation
    final_goals = int(cumulative_goals.iloc[-1])
    final_xg = cumulative_xg[-1]
    diff = final_goals - final_xg
    ax.annotate(
        f'Goals: {final_goals}\nxG: {final_xg:.1f}\nDiff: {diff:+.1f}',
        xy=(len(df_sorted), cumulative_goals.iloc[-1]),
        xytext=(10, 0),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    if save:
        if filename is None:
            safe_name = player_name.replace(' ', '_').lower()
            filename = f"cumulative_xg_{safe_name}.png"
        path = OUTPUT_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved cumulative xG plot to {path}")

    return fig


def griezmann_analysis(df, model=None, feature_columns=None):
    """
    Complete analysis for Antoine Griezmann.

    Returns summary and generates plots.
    """
    df_griezmann = get_player_shots(df, 'Griezmann')

    if len(df_griezmann) == 0:
        print("Could not find Griezmann in data")
        return None

    summary = player_summary(df_griezmann, model, feature_columns)
    print_player_summary(summary)

    # Generate plots
    plot_shot_map(df_griezmann, model, feature_columns,
                  filename="griezmann_shot_map.png")
    plot_cumulative_xg(df_griezmann, model, feature_columns,
                       filename="griezmann_cumulative_xg.png")

    return summary
