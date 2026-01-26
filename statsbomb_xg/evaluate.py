"""
Evaluation metrics and visualization for xG model.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    roc_curve
)
from sklearn.calibration import calibration_curve

from .config import OUTPUT_DIR, TRAIN_END_DATE


def temporal_split(df, cutoff=None):
    """
    Split data temporally for honest evaluation.

    Train on earlier matches, test on later matches.
    This prevents data leakage from future information.

    Args:
        df: DataFrame with 'match_date' column
        cutoff: Date string for train/test split (default: 2016-01-01)

    Returns:
        df_train, df_test
    """
    if cutoff is None:
        cutoff = TRAIN_END_DATE

    cutoff_date = pd.to_datetime(cutoff)

    df_train = df[df['match_date'] < cutoff_date].copy()
    df_test = df[df['match_date'] >= cutoff_date].copy()

    print(f"Temporal split at {cutoff}:")
    print(f"  Train: {len(df_train)} shots ({df_train['is_goal'].sum()} goals)")
    print(f"  Test:  {len(df_test)} shots ({df_test['is_goal'].sum()} goals)")

    return df_train, df_test


def evaluate_model(y_true, y_pred, sb_xg=None, label="Model"):
    """
    Calculate evaluation metrics.

    Args:
        y_true: Actual outcomes (0/1)
        y_pred: Predicted probabilities
        sb_xg: StatsBomb xG for comparison (optional)
        label: Label for printing

    Returns:
        dict of metrics
    """
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred),
        'log_loss': log_loss(y_true, y_pred),
        'n_shots': len(y_true),
        'n_goals': int(y_true.sum()),
        'total_xg': float(y_pred.sum()),
    }

    print(f"\n{label} Metrics:")
    print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Brier:     {metrics['brier']:.4f}")
    print(f"  Log Loss:  {metrics['log_loss']:.4f}")
    print(f"  Shots:     {metrics['n_shots']}")
    print(f"  Goals:     {metrics['n_goals']}")
    print(f"  Total xG:  {metrics['total_xg']:.1f}")

    if sb_xg is not None:
        sb_metrics = {
            'sb_roc_auc': roc_auc_score(y_true, sb_xg),
            'sb_brier': brier_score_loss(y_true, sb_xg),
            'sb_log_loss': log_loss(y_true, sb_xg),
            'sb_total_xg': float(sb_xg.sum()),
        }
        metrics.update(sb_metrics)

        print(f"\nStatsBomb Comparison:")
        print(f"  ROC AUC:   {sb_metrics['sb_roc_auc']:.4f} (gap: {sb_metrics['sb_roc_auc'] - metrics['roc_auc']:.4f})")
        print(f"  Brier:     {sb_metrics['sb_brier']:.4f}")
        print(f"  Total xG:  {sb_metrics['sb_total_xg']:.1f}")

    return metrics


def plot_calibration_curve(y_true, y_pred, sb_xg=None, n_bins=10, save=True):
    """
    Plot calibration curve comparing model to StatsBomb.

    A well-calibrated model should have points close to the diagonal.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Our model
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy='uniform')
    ax.plot(prob_pred, prob_true, 's-', label='Our Model', markersize=8)

    # StatsBomb
    if sb_xg is not None:
        sb_prob_true, sb_prob_pred = calibration_curve(y_true, sb_xg, n_bins=n_bins, strategy='uniform')
        ax.plot(sb_prob_pred, sb_prob_true, 'o-', label='StatsBomb', markersize=8)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save:
        path = OUTPUT_DIR / "calibration_curve.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved calibration curve to {path}")

    return fig


def plot_roc_curve(y_true, y_pred, sb_xg=None, save=True):
    """Plot ROC curve comparing model to StatsBomb."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Our model
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    ax.plot(fpr, tpr, label=f'Our Model (AUC = {auc:.3f})')

    # StatsBomb
    if sb_xg is not None:
        sb_fpr, sb_tpr, _ = roc_curve(y_true, sb_xg)
        sb_auc = roc_auc_score(y_true, sb_xg)
        ax.plot(sb_fpr, sb_tpr, label=f'StatsBomb (AUC = {sb_auc:.3f})')

    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        path = OUTPUT_DIR / "roc_curve.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to {path}")

    return fig


def plot_xg_delta_histogram(y_pred, sb_xg, save=True):
    """
    Plot histogram of shot-by-shot xG differences (Our - StatsBomb).

    This shows where our model systematically differs from StatsBomb.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    delta = y_pred - sb_xg

    ax.hist(delta, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', label='No difference')
    ax.axvline(x=delta.mean(), color='green', linestyle='-',
               label=f'Mean: {delta.mean():.4f}')

    ax.set_xlabel('xG Difference (Our Model - StatsBomb)')
    ax.set_ylabel('Number of Shots')
    ax.set_title('Shot-by-Shot xG Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        path = OUTPUT_DIR / "xg_delta_histogram.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved xG delta histogram to {path}")

    return fig


def plot_spatial_xg(df, model, feature_columns, save=True):
    """
    Create heatmap of xG values across the pitch.

    Shows how xG varies by shot location.
    """
    from .features import engineer_features

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create grid of shot locations
    x_range = np.linspace(60, 120, 30)  # Only attacking half
    y_range = np.linspace(0, 80, 20)
    xx, yy = np.meshgrid(x_range, y_range)

    # Create fake shots at each grid point
    grid_shots = pd.DataFrame({
        'x': xx.ravel(),
        'y': yy.ravel(),
        'body_part': 'Right Foot',
        'shot_type': 'Open Play',
        'play_pattern': 'Regular Play',
        'shot_first_time': False,
        'under_pressure': 0,
        'freeze_frame': [[] for _ in range(len(xx.ravel()))],
    })

    # Engineer features
    grid_shots = engineer_features(grid_shots)

    # Predict xG
    X_grid = grid_shots[feature_columns]
    xg_grid = model.predict_proba(X_grid)[:, 1]

    # Reshape for heatmap
    xg_heatmap = xg_grid.reshape(xx.shape)

    # Plot heatmap
    im = ax.contourf(xx, yy, xg_heatmap, levels=20, cmap='Reds')
    plt.colorbar(im, ax=ax, label='xG')

    # Draw pitch markings (simplified)
    # Goal
    ax.plot([120, 120], [36, 44], 'k-', linewidth=3)
    # 6-yard box
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], 'k-', linewidth=1)
    # 18-yard box
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], 'k-', linewidth=1)
    # Penalty spot
    ax.plot(108, 40, 'ko', markersize=5)

    ax.set_xlabel('X (yards)')
    ax.set_ylabel('Y (yards)')
    ax.set_title('Spatial xG Distribution (Open Play, Right Foot)')
    ax.set_xlim([60, 122])
    ax.set_ylim([0, 80])
    ax.set_aspect('equal')

    if save:
        path = OUTPUT_DIR / "spatial_xg.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved spatial xG plot to {path}")

    return fig


def save_metrics(metrics, filename="metrics.json"):
    """Save metrics to JSON file."""
    import json

    path = OUTPUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")
    return path
