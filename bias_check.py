import pandas as pd
import joblib

data = pd.read_csv("data/Maternal Health Risk Data Set.csv")

model = joblib.load("models/risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")

X = data.drop("RiskLevel", axis=1)
y = data["RiskLevel"]

X_scaled = scaler.transform(X)

preds = model.predict(X_scaled)

data["Prediction"] = preds

# Age groups
data["AgeGroup"] = pd.cut(
    data["Age"],
    bins=[15,20,30,40,50],
    labels=["Teen","Young","Adult","Advanced"]
)

bias_report = data.groupby("AgeGroup")["Prediction"].value_counts(normalize=True)

print("\nBias Audit Report\n")
print(bias_report)