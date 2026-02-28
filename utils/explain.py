import shap
import joblib
import numpy as np
from pathlib import Path

# -------------------------------------------------
# LOAD MODEL ARTIFACTS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "risk_model.pkl")
feature_names = joblib.load(MODEL_DIR / "features.pkl")

explainer = shap.TreeExplainer(model)


# -------------------------------------------------
# LOCAL EXPLAINABILITY
# -------------------------------------------------
def explain_prediction(input_data):

    shap_values = explainer.shap_values(input_data)

    pred_class = model.predict(input_data)[0]

    # multiclass handling
    if isinstance(shap_values, list):
        values = shap_values[pred_class][0]
    else:
        values = shap_values[0]

    values = np.ravel(values)

    importance = list(zip(feature_names, values))

    # sort by absolute influence
    importance_sorted = sorted(
        importance,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # ✅ PRIMARY RISK DRIVER = feature increasing risk
    positive_features = [f for f in importance if f[1] > 0]

    if positive_features:
        main_factor = max(
            positive_features,
            key=lambda x: x[1]
        )[0]
    else:
        main_factor = "Vitals Stable"

    return importance_sorted, main_factor