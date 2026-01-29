"""Pre-compute SHAP values and save to disk for faster app loading."""
import numpy as np
import shap
from pathlib import Path

from .config import MODEL_DIR, FEATURE_COLUMNS
from .data import load_shots
from .features import engineer_features
from .model import load_model


def precompute_and_save_shap():
    """Compute SHAP values and save to disk."""
    print("Loading data...")
    shots_df = load_shots()
    df = engineer_features(shots_df)

    print("Loading model...")
    model = load_model()

    print("Preparing test data...")
    test_df = df[df['match_date'] >= '2016-01-01']
    X_test = test_df[FEATURE_COLUMNS]

    print("Computing SHAP values...")
    # Extract base XGBoost estimator from CalibratedClassifierCV
    base_model = model.calibrated_classifiers_[0].estimator

    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(np.asarray(X_test))

    # Save SHAP values and feature data
    shap_path = MODEL_DIR / 'shap_values.npz'
    print(f"Saving SHAP values to {shap_path}...")
    np.savez_compressed(
        shap_path,
        shap_values=shap_values,
        feature_data=np.asarray(X_test),
        feature_names=FEATURE_COLUMNS
    )

    print("Done!")
    return shap_path


if __name__ == '__main__':
    precompute_and_save_shap()
