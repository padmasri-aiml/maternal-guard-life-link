# ======================================================
# Maternal-Guard & Life-Link
# ML Risk Predictor Training Pipeline (FINAL VERSION)
# ======================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ======================================================
# PATH CONFIGURATION
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Maternal Health Risk Data Set.csv"

MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

REPORT_FILE = REPORT_DIR / "training_results.txt"

log = open(REPORT_FILE, "w", encoding="utf-8")

def write_log(text):
    print(text)
    log.write(text + "\n")


# ======================================================
# LOAD DATA
# ======================================================

write_log("Loading dataset...")
df = pd.read_csv(DATA_PATH)

write_log(f"Dataset Shape: {df.shape}")
write_log(f"Columns: {list(df.columns)}")


# ======================================================
# OUTLIER HANDLING (Problem Requirement)
# ======================================================

def clip_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower, upper)

df["Age"] = clip_outliers(df["Age"])
df["HeartRate"] = clip_outliers(df["HeartRate"])

write_log("Outliers handled using IQR clipping.")


# ======================================================
# LABEL ENCODING
# ======================================================

le = LabelEncoder()
df["RiskLevel"] = le.fit_transform(df["RiskLevel"])

X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

feature_names = X.columns.tolist()


# ======================================================
# FEATURE SCALING
# (Needed mainly for SVM but kept consistent)
# ======================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ======================================================
# MODELS (Problem Statement Requirement)
# ======================================================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "SVM": SVC(kernel="rbf", probability=True),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=42
    )
}


# ======================================================
# STRATIFIED CROSS VALIDATION
# ======================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

write_log("\n===== Stratified Cross Validation =====")

for name, model in models.items():

    write_log(f"\nModel: {name}")

    fold_scores = []
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f1 = f1_score(y_test, preds, average="macro")
        fold_scores.append(f1)

        all_true.extend(y_test)
        all_pred.extend(preds)

        write_log(f"Fold {fold+1} F1: {f1:.4f}")

    mean_f1 = np.mean(fold_scores)
    write_log(f"Mean F1 Score: {mean_f1:.4f}")

    results.append([name, mean_f1])

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(all_true, all_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low","Mid","High"],
        yticklabels=["Low","Mid","High"]
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"confusion_matrix_{name}.png")
    plt.close()

    write_log("\nClassification Report:")
    write_log(classification_report(all_true, all_pred))


# ======================================================
# MODEL COMPARISON
# ======================================================

results_df = pd.DataFrame(results, columns=["Model","F1"])
results_df.to_csv(REPORT_DIR / "model_metrics.csv", index=False)

plt.figure(figsize=(6,4))
sns.barplot(x="Model", y="F1", data=results_df)
plt.title("Model Comparison (Macro F1 — Stratified CV)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "model_comparison.png")
plt.close()


# ======================================================
# SELECT BEST MODEL
# ======================================================

best_model_name = results_df.sort_values(
    "F1", ascending=False
).iloc[0]["Model"]

write_log(f"\n✅ BEST MODEL: {best_model_name}")

best_model = models[best_model_name]


# ======================================================
# ⭐ IMPORTANT FIX
# RETRAIN BEST MODEL ON FULL DATASET
# ======================================================

write_log("\nRetraining best model on FULL dataset...")
best_model.fit(X_scaled, y)


# ======================================================
# GLOBAL FEATURE IMPORTANCE (Explainability Requirement)
# ======================================================

if hasattr(best_model, "feature_importances_"):

    importance = best_model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    imp_df.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(7,4))
    sns.barplot(x="Importance", y="Feature", data=imp_df)
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_importance.png")
    plt.close()

    write_log("\nFeature Importance:")
    write_log(str(imp_df))


# ======================================================
# SAVE MODEL ARTIFACTS
# ======================================================

joblib.dump(best_model, MODEL_DIR / "risk_model.pkl")
joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(feature_names, MODEL_DIR / "features.pkl")


# ======================================================
# FINAL SUMMARY
# ======================================================

write_log("\n================ FINAL MODEL SUMMARY ================")

best_row = results_df.sort_values("F1", ascending=False).iloc[0]

write_log(f"Selected Model : {best_row['Model']}")
write_log(f"Macro F1 Score : {best_row['F1']:.4f}")

write_log("""
Model Selection Justification:
- Stratified Cross Validation used to address class imbalance.
- Macro F1-score ensures balanced evaluation.
- Final model retrained on full dataset for stability.
""")

write_log("=====================================================")

log.close()

print("\n✅ Training completed successfully.")
print("Reports saved in /reports")
print("Plots saved in /plots")