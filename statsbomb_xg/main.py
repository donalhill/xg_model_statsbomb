"""
Main pipeline for xG model training and evaluation.

Run this script to:
1. Load and cache StatsBomb data
2. Engineer features
3. (Optional) Optimize hyperparameters with Optuna
4. Train XGBoost model with isotonic calibration
5. Evaluate with temporal validation
6. Generate plots and metrics
7. Run Griezmann analysis

Usage:
    python -m statsbomb_xg.main                    # Use default/cached params
    python -m statsbomb_xg.main --optimize         # Run Optuna optimization
    python -m statsbomb_xg.main --optimize --trials 200  # More trials
"""
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

from .config import FEATURE_COLUMNS, OUTPUT_DIR
from .data import load_shots
from .features import engineer_features
from .model import train_xg_model, predict_xg, save_model, get_feature_importance
from .evaluate import (
    temporal_split, evaluate_model, save_metrics,
    plot_calibration_curve, plot_roc_curve,
    plot_xg_delta_histogram, plot_spatial_xg
)
from .player_analysis import griezmann_analysis
from .optimize import optimize_hyperparameters, load_best_params, get_default_params
import matplotlib.pyplot as plt


def main(optimize=False, n_trials=100):
    """Run the full xG model pipeline."""
    print("="*60)
    print("xG Model Pipeline - La Liga 2015/16")
    print("="*60)

    # 1. Load data
    print("\n[1/8] Loading shot data...")
    df = load_shots()
    print(f"Total shots: {len(df)}")
    print(f"Total goals: {df['is_goal'].sum()} ({df['is_goal'].mean():.1%})")

    # 2. Engineer features
    print("\n[2/8] Engineering features...")
    df = engineer_features(df)
    print(f"Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS[:5]}...")

    # 3. Temporal split
    print("\n[3/8] Temporal train/test split...")
    df_train, df_test = temporal_split(df)

    X_train = df_train[FEATURE_COLUMNS]
    y_train = df_train['is_goal']
    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test['is_goal']
    sb_test = df_test['statsbomb_xg']

    # 4. Get hyperparameters (optimize or load cached)
    print("\n[4/8] Hyperparameter selection...")
    if optimize:
        # Run Optuna optimization on training data
        params = optimize_hyperparameters(
            X_train, y_train,
            dates=df_train['match_date'],
            n_trials=n_trials
        )
    else:
        # Try to load previously optimized params
        params = load_best_params()
        if params:
            print("Using previously optimized parameters")
        else:
            print("Using default parameters (run with --optimize to tune)")
            params = get_default_params()

    print(f"Parameters: {params}")

    # 5. Train final model on full training set
    print("\n[5/8] Training XGBoost with isotonic calibration...")
    model = train_xg_model(X_train, y_train, params=params, calibrate=True)

    # Get predictions
    y_pred = predict_xg(model, X_test)

    # 6. Evaluate
    print("\n[6/8] Evaluating model...")
    metrics = evaluate_model(y_test, y_pred, sb_xg=sb_test, label="Test Set")

    # Feature importance
    importance_df = get_feature_importance(model, FEATURE_COLUMNS)
    print("\nFeature Importance (top 10):")
    print(importance_df.head(10).to_string(index=False))

    # 7. Generate plots
    print("\n[7/8] Generating plots...")
    plot_calibration_curve(y_test, y_pred, sb_xg=sb_test)
    plot_roc_curve(y_test, y_pred, sb_xg=sb_test)
    plot_xg_delta_histogram(y_pred, sb_test)
    plot_spatial_xg(df_test, model, FEATURE_COLUMNS)
    plt.close('all')  # Close all figures to free memory

    # 8. Player analysis
    print("\n[8/8] Griezmann analysis...")
    griezmann_summary = griezmann_analysis(df, model, FEATURE_COLUMNS)
    plt.close('all')

    # Save outputs
    print("\nSaving outputs...")
    save_model(model)

    # Add feature importance and params to metrics
    importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
    metrics['feature_importance'] = importance_dict
    metrics['hyperparameters'] = params

    # Add Griezmann summary
    if griezmann_summary:
        metrics['griezmann'] = griezmann_summary

    save_metrics(metrics)

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nKey results:")
    print(f"  - ROC AUC: {metrics['roc_auc']:.4f} (StatsBomb: {metrics.get('sb_roc_auc', 0):.4f})")
    print(f"  - Brier:   {metrics['brier']:.4f} (StatsBomb: {metrics.get('sb_brier', 0):.4f})")

    if griezmann_summary:
        print(f"\nGriezmann:")
        print(f"  - Shots: {griezmann_summary['total_shots']}")
        print(f"  - Goals: {griezmann_summary['total_goals']}")
        print(f"  - Our xG: {griezmann_summary.get('our_xg', 0):.1f}")
        print(f"  - Overperformance: {griezmann_summary.get('goals_minus_our_xg', 0):+.1f}")

    print("\nTo launch dashboard: python -m statsbomb_xg.app")

    return model, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xG Model Pipeline')
    parser.add_argument('--optimize', action='store_true',
                        help='Run Optuna hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of Optuna trials (default: 100)')
    args = parser.parse_args()

    main(optimize=args.optimize, n_trials=args.trials)
