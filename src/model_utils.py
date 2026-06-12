from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_model_bundle(models_dir: Path | str = MODELS_DIR) -> tuple[Any, dict]:
    """Load the trained model and metadata saved by train_model.py/notebook."""
    models_dir = Path(models_dir)
    model_path = models_dir / "xgb_demand_model.pkl"
    meta_path = models_dir / "model_features.pkl"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Model artifacts were not found. Run train_model.ipynb or train_model.py first."
        )

    model = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    return model, metadata


def prepare_prediction_input(user_inputs: dict, metadata: dict) -> pd.DataFrame:
    """Build a single-row feature frame aligned to the trained model."""
    feature_names = metadata["feature_names"]
    defaults = metadata.get("feature_defaults", {})
    row = {feature: defaults.get(feature, 0) for feature in feature_names}

    for key, value in user_inputs.items():
        if key in row:
            row[key] = value

    selected_type = user_inputs.get("Type")
    if selected_type:
        for feature in feature_names:
            if feature.startswith("Type_"):
                row[feature] = 1 if feature == f"Type_{selected_type}" else 0

    selected_season = user_inputs.get("Season")
    if selected_season:
        for feature in feature_names:
            if feature.startswith("Season_"):
                row[feature] = 1 if feature == f"Season_{selected_season}" else 0

    return pd.DataFrame([row], columns=feature_names)


def predict_weekly_sales(user_inputs: dict, models_dir: Path | str = MODELS_DIR) -> float:
    """Predict weekly sales from app/user inputs."""
    model, metadata = load_model_bundle(models_dir)
    X = prepare_prediction_input(user_inputs, metadata)
    prediction = float(model.predict(X)[0])
    return max(0.0, prediction)


def estimate_inventory_risk(predicted_demand: float, current_inventory_value: float) -> dict:
    """Estimate stockout/overstock risk using forecasted demand and user-provided inventory value."""
    if current_inventory_value < predicted_demand * 0.7:
        risk = "High Stockout Risk"
    elif current_inventory_value > predicted_demand * 2.0:
        risk = "High Overstock Risk"
    else:
        risk = "Low Risk"

    reorder_value = max(0.0, predicted_demand * 1.2 - current_inventory_value)
    return {
        "risk": risk,
        "reorder_value": reorder_value,
        "coverage_ratio": current_inventory_value / predicted_demand if predicted_demand else 0,
    }
