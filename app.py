# ==========================================================
# Maternal-Guard & Life-Link
# Main Application (FINAL VERSION)
# ==========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from predict import predict_risk
from donor_match import match_donors
from utils.security import mask_medical_data


# ----------------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------------
st.set_page_config(
    page_title="Maternal-Guard & Life-Link",
    layout="wide"
)

st.title("🩺 Maternal-Guard & Life-Link")
st.subheader("AI-Driven Maternal Risk Monitoring")

st.info("Enter patient vitals on the left and click **Predict Risk**.")

# ----------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "alert_triggered" not in st.session_state:
    st.session_state.alert_triggered = False

if "donors" not in st.session_state:
    st.session_state.donors = None


# ----------------------------------------------------------
# SIDEBAR — PATIENT INPUT
# ----------------------------------------------------------
with st.sidebar:

    st.header("Patient Vitals")

    age = st.number_input("Age", 15, 50, 25)

    blood_group = st.selectbox(
        "Patient Blood Group",
        ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    )

    systolic = st.number_input("Systolic BP", 80, 200, 120)
    diastolic = st.number_input("Diastolic BP", 40, 130, 80)
    bs = st.number_input("Blood Sugar", 3.0, 20.0, 6.0)
    temp = st.number_input("Body Temperature", 34.0, 42.0, 36.8)
    hr = st.number_input("Heart Rate", 40, 180, 75)

    predict_btn = st.button("🔎 Predict Risk")


# ----------------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------------
col_center, col_graph = st.columns([2, 1])


# ==========================================================
# RUN PREDICTION
# ==========================================================
if predict_btn:

    features = [age, systolic, diastolic, bs, temp, hr]

    risk, importance, main_factor = predict_risk(features)

    st.session_state.prediction_done = True
    st.session_state.risk = risk
    st.session_state.importance = importance
    st.session_state.main_factor = main_factor

    # reset emergency flow
    st.session_state.alert_triggered = False
    st.session_state.donors = None


# ==========================================================
# DISPLAY RESULTS
# ==========================================================
if st.session_state.prediction_done:

    risk = st.session_state.risk
    importance = st.session_state.importance
    main_factor = st.session_state.main_factor

    # ---------- CENTER PANEL ----------
    with col_center:

        st.header("🧠 Risk Assessment")

        color_map = {
            "low risk": "green",
            "mid risk": "orange",
            "high risk": "red"
        }

        st.markdown(
            f"<h2 style='color:{color_map.get(risk.lower(),'black')}'>"
            f"Risk Level: {risk}</h2>",
            unsafe_allow_html=True
        )

        st.success(f"Primary Risk Driver: {main_factor}")

    # ---------- GRAPH PANEL ----------
    with col_graph:

        st.header("📊 Influencing Factors")

        imp_df = pd.DataFrame(
            importance.items(),
            columns=["Feature", "Importance"]
        ).sort_values("Importance")

        fig, ax = plt.subplots(figsize=(4, 2.8))

        ax.barh(
            imp_df["Feature"],
            imp_df["Importance"]
        )

        ax.set_xlabel("Influence Strength")
        ax.set_ylabel("")

        plt.tight_layout()
        st.pyplot(fig)


# ==========================================================
# EMERGENCY DONOR DISPATCH
# ==========================================================
if (
    st.session_state.prediction_done
    and "high" in st.session_state.risk.lower()
):

    st.markdown("---")
    st.header("🚨 Emergency Donor Dispatch")

    if not st.session_state.alert_triggered:

        if st.button("Activate Hemorrhage Alert"):

            donors = match_donors(
                blood_group,
                16.5062,   # Vijayawada reference
                80.6480
            )

            st.session_state.donors = donors
            st.session_state.alert_triggered = True

    # Show donor results
    if st.session_state.donors is not None:

        donors = st.session_state.donors.copy()

        if donors.empty:
            st.warning("No compatible donors available.")
        else:
            st.success("Compatible donors identified.")

            # 🔐 MASK MEDICAL DATA
            donors["chronic_conditions"] = donors[
                "chronic_conditions"
            ].apply(mask_medical_data)

            st.caption(
                "🔒 Donor medical information is encrypted "
                "and hidden for privacy protection."
            )

            st.dataframe(donors, use_container_width=True)


# ==========================================================
# ETHICAL AI & CONSENT VISIBILITY
# ==========================================================
st.markdown("---")
st.header("🔐 Ethical AI & Consent Compliance")

st.success(
"""
✔ Donor Consent Management implemented via **admin_donor_manager.py**

✔ Bias Mitigation validated using **bias_check.py**
(age-group fairness evaluation)

✔ Donor medical histories are encrypted and privacy-masked.
"""
)