# ======================================================
# Prediction Module
# ======================================================

import joblib
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "risk_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
feature_names = joblib.load(MODEL_DIR / "features.pkl")


# ------------------------------------------------------
# Predict Risk + Feature Importance
# ------------------------------------------------------
def predict_risk(features):

    arr = np.array(features).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    pred = model.predict(arr_scaled)[0]
    risk_label = label_encoder.inverse_transform([pred])[0]

    # GLOBAL FEATURE IMPORTANCE (RandomForest)
    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_

        # Convert to dictionary (IMPORTANT FIX)
        importance = dict(
            zip(feature_names, importance_values)
        )

        main_factor = max(importance, key=importance.get)

    else:
        importance = {}
        main_factor = "N/A"

    return risk_label, importance, main_factor