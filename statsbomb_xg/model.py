"""
XGBoost model training with isotonic calibration.
"""
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

from .config import MODEL_DIR


def train_xg_model(X_train, y_train, params=None, calibrate=True):
    """
    Train XGBoost model with optional isotonic calibration.

    NO class weights - we want calibrated P(goal), not better recall.
    Class weights would artificially inflate probabilities for the minority
    class, destroying our probability calibration.

    Isotonic calibration: non-parametric mapping of raw probabilities to
    calibrated probabilities. Learns a monotonic function that maps
    predicted probs to observed frequencies.

    Args:
        X_train: Training features
        y_train: Training labels (0/1)
        params: XGBoost hyperparameters (dict). If None, uses defaults.
        calibrate: Whether to apply isotonic calibration

    Returns:
        Trained model (calibrated or raw XGBoost)
    """
    # Default parameters (can be overridden by Optuna-tuned params)
    default_params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    }

    if params is not None:
        default_params.update(params)

    xgb_model = XGBClassifier(
        **default_params,
        random_state=42,
        eval_metric='logloss',
    )

    if calibrate:
        # CalibratedClassifierCV with isotonic regression
        # cv=5 means 5-fold cross-validation for calibration
        model = CalibratedClassifierCV(
            xgb_model,
            method='isotonic',
            cv=5
        )
    else:
        model = xgb_model

    print(f"Training {'calibrated ' if calibrate else ''}XGBoost model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    return model


def predict_xg(model, X):
    """
    Get xG predictions from trained model.

    Args:
        model: Trained model
        X: Features to predict on

    Returns:
        Array of xG values (probabilities)
    """
    # Convert to numpy array to avoid XGBoost 3.x feature name mismatch issues
    import numpy as np
    X_arr = np.asarray(X)
    return model.predict_proba(X_arr)[:, 1]


def save_model(model, filename="xg_model.joblib"):
    """Save model to disk."""
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path


def load_model(filename="xg_model.joblib"):
    """Load model from disk."""
    path = MODEL_DIR / filename
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from XGBoost model.

    Works with both raw XGBoost and CalibratedClassifierCV.
    """
    # For CalibratedClassifierCV, extract base estimator
    if hasattr(model, 'calibrated_classifiers_'):
        # Get importance from first fold's base estimator
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model

    importance = base_model.feature_importances_

    # Create sorted dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df
