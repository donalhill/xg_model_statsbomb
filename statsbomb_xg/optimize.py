"""
Hyperparameter optimization using Optuna.

Professional approach:
- Optimize Brier score (proper scoring rule)
- Temporal cross-validation (no future leakage)
- Tune full pipeline including calibration
"""
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from .config import OUTPUT_DIR

# Suppress Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def temporal_cv_brier(params, X, y, dates, n_splits=5):
    """
    Evaluate model using temporal cross-validation.

    Returns mean Brier score across folds.
    Uses CalibratedClassifierCV within each fold.
    """
    # Sort by date for proper temporal splits
    sort_idx = np.argsort(dates)
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    brier_scores = []

    for train_idx, val_idx in tscv.split(X_sorted):
        X_train, X_val = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
        y_train, y_val = y_sorted.iloc[train_idx], y_sorted.iloc[val_idx]

        # Skip if validation set is too small or has no positive class
        if len(y_val) < 50 or y_val.sum() < 5:
            continue

        # Train XGBoost with calibration
        xgb = XGBClassifier(**params, random_state=42, eval_metric='logloss')

        # Calibrate with 3-fold CV (smaller than default due to fold size)
        calibrated = CalibratedClassifierCV(xgb, method='isotonic', cv=3)

        try:
            calibrated.fit(X_train, y_train)
            y_pred = calibrated.predict_proba(X_val)[:, 1]
            brier = brier_score_loss(y_val, y_pred)
            brier_scores.append(brier)
        except Exception:
            # Skip fold if calibration fails (can happen with small folds)
            continue

    if not brier_scores:
        return 1.0  # Worst possible score

    return np.mean(brier_scores)


def objective(trial, X, y, dates):
    """Optuna objective function."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }

    return temporal_cv_brier(params, X, y, dates)


def optimize_hyperparameters(X, y, dates, n_trials=100, show_progress=True):
    """
    Run Optuna optimization to find best hyperparameters.

    Args:
        X: Feature matrix
        y: Target variable
        dates: Match dates for temporal ordering
        n_trials: Number of optimization trials
        show_progress: Whether to show progress bar

    Returns:
        dict: Best hyperparameters
    """
    print(f"Running Optuna optimization ({n_trials} trials)...")
    print("Objective: Minimize Brier score with temporal CV")

    study = optuna.create_study(
        direction='minimize',
        study_name='xg_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective(trial, X, y, dates),
        n_trials=n_trials,
        show_progress_bar=show_progress,
        n_jobs=1  # XGBoost handles parallelism internally
    )

    print(f"\nBest Brier score (CV): {study.best_value:.5f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save study results
    save_optimization_results(study)

    return study.best_params


def save_optimization_results(study):
    """Save optimization history and best params."""
    import json

    results = {
        'best_brier': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
    }

    path = OUTPUT_DIR / 'optimization_results.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved optimization results to {path}")


def get_default_params():
    """Return sensible default params (used if optimization is skipped)."""
    return {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    }


def load_best_params():
    """Load previously optimized params if available."""
    import json

    path = OUTPUT_DIR / 'optimization_results.json'
    if path.exists():
        with open(path) as f:
            results = json.load(f)
        return results['best_params']
    return None
