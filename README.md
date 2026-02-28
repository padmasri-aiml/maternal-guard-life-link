🩺 Maternal-Guard & Life-Link
AI-Driven Maternal Care & Emergency Donor Network

🔗 Live Application:
https://maternal-guard-life-link-7zelkrtkvpyaackvrh3ddc.streamlit.app/

Maternal-Guard & Life-Link is an AI-powered healthcare decision-support system designed to assist rural healthcare workers in identifying high-risk maternal cases and rapidly connecting compatible blood donors during postpartum emergencies.

The system transforms maternal healthcare from reactive treatment into proactive AI-assisted monitoring using Machine Learning and an emergency donor dispatch network.

🚨 Problem Statement

In rural healthcare environments:

High-risk pregnancies are detected late.

Clinical assessment depends heavily on manual judgment.

Blood donor coordination is slow and fragmented.

Postpartum hemorrhage requires immediate response during the golden hour.

Delays in risk identification and donor availability significantly increase maternal mortality risk.

💡 Proposed Solution

Maternal-Guard & Life-Link integrates:

1️⃣ AI Maternal Risk Predictor

Predicts maternal risk using patient vitals.

2️⃣ Life-Link Emergency Donor Network

Instantly identifies compatible donors during emergencies.

Workflow

Patient Vitals → AI Risk Predictor → Risk Level
↓
Hemorrhage Alert
↓
Compatible Donor Identification

🏗️ System Architecture
Component A — ML Risk Predictor

Input Features:

Age

Systolic Blood Pressure

Diastolic Blood Pressure

Blood Sugar

Body Temperature

Heart Rate

Blood Group


Models Evaluated:

Random Forest

Support Vector Machine (SVM)

XGBoost

Selected Model:
Random Forest (Best Macro F1 Score)

Explainability:
Feature Importance identifies primary clinical risk drivers.

Component B — Life-Link Donor Network

SQLite donor database

Blood group compatibility matching

Hemoglobin eligibility filtering (>12.5 g/dL)

Donor availability (consent management)

Encrypted medical records

⚙️ Technical Stack

Frontend: Streamlit
Backend: Python
Machine Learning: Scikit-Learn, XGBoost
Database: SQLite
Explainable AI: Feature Importance
Security: Fernet Encryption
Visualization: Matplotlib
Deployment: Streamlit Community Cloud
Version Control: GitHub

🤖 Machine Learning Pipeline
Preprocessing

Outlier handling using IQR clipping

Feature scaling (StandardScaler)

Label encoding of risk categories

Validation

Stratified 5-Fold Cross Validation

Handles class imbalance

📊 Model Performance

Macro F1 Score Comparison:

Random Forest — 0.861
XGBoost — 0.806
SVM — 0.688

Selected Model:
Random Forest provides balanced performance across all maternal risk categories and reliable high-risk detection.

📈 Model Report

Generated using:

python train_model.py

Outputs include:

F1 Score

Classification Report

Confusion Matrix

Feature Importance visualization

Saved inside:

reports/

plots/

🔍 Explainable AI

The system explains predictions using:

Global Feature Importance from Random Forest

Identification of primary influencing vital parameter

This increases clinical trust and interpretability.

🚑 Emergency Donor Dispatch

When a patient is classified as High Risk:

Healthcare worker activates Hemorrhage Alert.

System filters donors by:

Blood compatibility

Hemoglobin ≥ 12.5 g/dL

Availability status

Compatible donors are displayed instantly.

🔐 Security & Ethical Design
Data Protection

Medical histories encrypted using Fernet encryption.

Sensitive data masked in the interface.

Consent Management

Implemented via admin donor console allowing to mark unavailable without deleting profiles.

Run:
python admin_donor_manager.py

Bias Mitigation

Age-group fairness evaluated using prediction distribution analysis.

Run:
python bias_check.py

📂 Complete Project Structure

maternal_guard_project/

app.py
→ Main Streamlit web application (user interface)

train_model.py
→ Trains ML models, performs cross-validation, generates reports & plots

predict.py
→ Loads trained model and performs real-time prediction

donor_match.py
→ Blood compatibility logic and donor filtering

init_donor_db.py
→ Creates and populates donor database

admin_donor_manager.py
→ Consent management (update donor availability)

bias_check.py
→ Bias auditing across age groups

models/
→ Saved trained model, scaler, encoder

data/
→ Maternal Health Risk dataset

plots/
→ Confusion matrix & feature importance graphs

reports/
→ Training results and evaluation metrics

utils/security.py
→ Encryption and privacy masking utilities

donors.db
→ SQLite donor database

▶️ How to Run Locally
1. Install Dependencies

pip install -r requirements.txt

2. Initialize Donor Database (First Time Only)

python init_donor_db.py

3. Train Machine Learning Model

python train_model.py

This generates:

trained model

evaluation reports

plots

4. Run Bias Audit (Optional)

python bias_check.py

5. Manage Donor Availability (Optional Admin Tool)

python admin_donor_manager.py

6. Launch Application

streamlit run app.py

Open browser at:
http://localhost:8501

🌍 Deployment

Live deployed application:

https://maternal-guard-life-link-7zelkrtkvpyaackvrh3ddc.streamlit.app/

Deployed using Streamlit Community Cloud for real-time demonstration.

🎯 Impact

Supports rural healthcare workers

Enables proactive maternal monitoring

Reduces emergency response time

Bridges hospitals and donor communities

Improves maternal emergency preparedness

🚀 Future Enhancements

Mobile health worker interface

IoT-based vital monitoring

District-scale donor ecosystem

Hospital system integration APIs

👩‍💻 Author

Padmasri
AI & Machine Learning — Datathon Submission

