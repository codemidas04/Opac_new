import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Load model
import os

@st.cache_resource
def load_model():
    possible_paths = [
        "credit_model.pkl",                  # root
        os.path.join("backend", "credit_model.pkl"),  # backend/
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("❌ credit_model.pkl not found in root or backend/. Please train the model first.")

model = load_model()

# Feature mapping (codes → meanings)
feature_name_map = {
    # ------------------------
    # Numerical features
    # ------------------------
    "num__Duration": "Loan Duration (months)",
    "num__CreditAmount": "Credit Amount",
    "num__InstallmentRate": "Installment Rate (% of income)",
    "num__Age": "Age (years)",
    "num__ResidenceDuration": "Residence Duration (years)",
    "num__ExistingCredits": "Number of Existing Credits",
    "num__LiableDependents": "Number of Dependents",
    # ------------------------
    # Status of Checking Account
    "cat__Status_A11": "Checking Account < 0 DM",
    "cat__Status_A12": "Checking Account 0 ≤ ... < 200 DM",
    "cat__Status_A13": "Checking Account ≥ 200 DM",
    "cat__Status_A14": "No Checking Account",
    # ------------------------
    # Credit History
    "cat__CreditHistory_A30": "No credit / all paid",
    "cat__CreditHistory_A31": "All credits paid (this bank)",
    "cat__CreditHistory_A32": "Existing credits paid properly",
    "cat__CreditHistory_A33": "Delay in past payments",
    "cat__CreditHistory_A34": "Critical / other credits",
    # ------------------------
    # Purpose
    "cat__Purpose_A40": "Car (new)",
    "cat__Purpose_A41": "Car (used)",
    "cat__Purpose_A42": "Furniture / Equipment",
    "cat__Purpose_A43": "Radio / Television",
    "cat__Purpose_A44": "Domestic appliances",
    "cat__Purpose_A45": "Repairs",
    "cat__Purpose_A46": "Education",
    "cat__Purpose_A47": "Vacation",
    "cat__Purpose_A48": "Retraining",
    "cat__Purpose_A49": "Business",
    "cat__Purpose_A410": "Other",
    # ------------------------
    # Savings
    "cat__Savings_A61": "Savings < 100 DM",
    "cat__Savings_A62": "Savings 100 ≤ ... < 500 DM",
    "cat__Savings_A63": "Savings 500 ≤ ... < 1000 DM",
    "cat__Savings_A64": "Savings ≥ 1000 DM",
    "cat__Savings_A65": "No savings account",
    # ------------------------
    # Employment
    "cat__Employment_A71": "Unemployed",
    "cat__Employment_A72": "Employment < 1 year",
    "cat__Employment_A73": "Employment 1–4 years",
    "cat__Employment_A74": "Employment 4–7 years",
    "cat__Employment_A75": "Employment ≥ 7 years",
    # ------------------------
    # Personal Status & Sex
    "cat__PersonalStatusSex_A91": "Male: divorced/separated",
    "cat__PersonalStatusSex_A92": "Female: divorced/separated/married",
    "cat__PersonalStatusSex_A93": "Male: single",
    "cat__PersonalStatusSex_A94": "Male: married/widowed",
    "cat__PersonalStatusSex_A95": "Female: single",
    # ------------------------
    # Other Debtors / Guarantors
    "cat__OtherDebtors_A101": "No other debtors",
    "cat__OtherDebtors_A102": "Co-applicant",
    "cat__OtherDebtors_A103": "Guarantor",
    # ------------------------
    # Property
    "cat__Property_A121": "Real estate",
    "cat__Property_A122": "Savings / Insurance",
    "cat__Property_A123": "Car or other assets",
    "cat__Property_A124": "Unknown / No property",
    # ------------------------
    # Other Installment Plans
    "cat__OtherInstallmentPlans_A141": "Bank",
    "cat__OtherInstallmentPlans_A142": "Stores",
    "cat__OtherInstallmentPlans_A143": "None",
    # ------------------------
    # Housing
    "cat__Housing_A151": "Rent",
    "cat__Housing_A152": "Own",
    "cat__Housing_A153": "Free",
    # ------------------------
    # Job
    "cat__Job_A171": "Unemployed / Unskilled (non-resident)",
    "cat__Job_A172": "Unskilled (resident)",
    "cat__Job_A173": "Skilled employee / official",
    "cat__Job_A174": "Management / Self-employed",
    # ------------------------
    # Telephone
    "cat__Telephone_A191": "No telephone",
    "cat__Telephone_A192": "Yes, registered",
    # ------------------------
    # Foreign Worker
    "cat__ForeignWorker_A201": "Yes",
    "cat__ForeignWorker_A202": "No",
}

# --- Streamlit UI ---
st.set_page_config(page_title="OpacGuard – Credit Risk", layout="wide")
st.title("💳 Credit Risk Prediction Dashboard")

# Tabs
tab1, tab2 = st.tabs(["🔮 Predict Credit Risk", "📊 Explain Prediction"])

# 🔹 Helper to build dropdowns automatically from feature_name_map
def get_options(prefix, codes):
    return {feature_name_map[f"cat__{prefix}_{code}"]: code for code in codes}

# ------------------------
# Predict Tab
# ------------------------
with tab1:
    # --- Threshold slider (new) ---
    threshold = st.sidebar.slider(
        "Decision Threshold (Bad Credit if probability ≥ threshold)",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05
    )

    with st.form("credit_form"):
        st.subheader("📋 Enter Applicant Information")

        col1, col2 = st.columns(2)

        with col1:
            duration = st.number_input("Loan Duration (months)", min_value=4, max_value=72, value=12)
            credit_amount = st.number_input("Credit Amount", min_value=250, max_value=20000, value=1000)
            age = st.number_input("Age (years)", min_value=18, max_value=75, value=30)
            installment_rate = st.number_input("Installment Rate (% of income)", min_value=1, max_value=4, value=2)
            residence_duration = st.number_input("Residence Duration (years)", min_value=1, max_value=4, value=2)
            existing_credits = st.number_input("Number of Existing Credits", min_value=1, max_value=4, value=1)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=2, value=0)

        with col2:
            status = st.selectbox("Status of Checking Account", list(get_options("Status", ["A11","A12","A13","A14"]).keys()))
            credit_history = st.selectbox("Credit History", list(get_options("CreditHistory", ["A30","A31","A32","A33","A34"]).keys()))
            purpose = st.selectbox("Purpose", list(get_options("Purpose", ["A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A410"]).keys()))
            savings = st.selectbox("Savings", list(get_options("Savings", ["A61","A62","A63","A64","A65"]).keys()))
            employment = st.selectbox("Employment", list(get_options("Employment", ["A71","A72","A73","A74","A75"]).keys()))
            personal_status = st.selectbox("Personal Status & Sex", list(get_options("PersonalStatusSex", ["A91","A92","A93","A94","A95"]).keys()))
            other_debtors = st.selectbox("Other Debtors / Guarantors", list(get_options("OtherDebtors", ["A101","A102","A103"]).keys()))
            property_ = st.selectbox("Property", list(get_options("Property", ["A121","A122","A123","A124"]).keys()))
            other_installments = st.selectbox("Other Installment Plans", list(get_options("OtherInstallmentPlans", ["A141","A142","A143"]).keys()))
            housing = st.selectbox("Housing", list(get_options("Housing", ["A151","A152","A153"]).keys()))
            job = st.selectbox("Job", list(get_options("Job", ["A171","A172","A173","A174"]).keys()))
            telephone = st.selectbox("Telephone", list(get_options("Telephone", ["A191","A192"]).keys()))
            foreign_worker = st.selectbox("Foreign Worker", list(get_options("ForeignWorker", ["A201","A202"]).keys()))

        # ------------------------
        # Submit button
        # ------------------------
        submitted = st.form_submit_button("🔮 Predict Credit Risk")

        if submitted:
            form_data = {
                "Duration": duration,
                "CreditAmount": credit_amount,
                "Age": age,
                "InstallmentRate": installment_rate,
                "ResidenceDuration": residence_duration,
                "ExistingCredits": existing_credits,
                "LiableDependents": dependents,
                "Status": get_options("Status", ["A11","A12","A13","A14"])[status],
                "CreditHistory": get_options("CreditHistory", ["A30","A31","A32","A33","A34"])[credit_history],
                "Purpose": get_options("Purpose", ["A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A410"])[purpose],
                "Savings": get_options("Savings", ["A61","A62","A63","A64","A65"])[savings],
                "Employment": get_options("Employment", ["A71","A72","A73","A74","A75"])[employment],
                "PersonalStatusSex": get_options("PersonalStatusSex", ["A91","A92","A93","A94","A95"])[personal_status],
                "OtherDebtors": get_options("OtherDebtors", ["A101","A102","A103"])[other_debtors],
                "Property": get_options("Property", ["A121","A122","A123","A124"])[property_],
                "OtherInstallmentPlans": get_options("OtherInstallmentPlans", ["A141","A142","A143"])[other_installments],
                "Housing": get_options("Housing", ["A151","A152","A153"])[housing],
                "Job": get_options("Job", ["A171","A172","A173","A174"])[job],
                "Telephone": get_options("Telephone", ["A191","A192"])[telephone],
                "ForeignWorker": get_options("ForeignWorker", ["A201","A202"])[foreign_worker],
            }

            df = pd.DataFrame([form_data])
            proba = model.predict_proba(df)[0][1]   # probability of Bad Credit
            prediction = 1 if proba >= threshold else 0
            label = "Good Credit ✅" if prediction == 0 else "Bad Credit ❌"

            # Save in session_state for Explain tab
            st.session_state["form_data"] = form_data
            st.session_state["prediction"] = label
            st.session_state["proba"] = proba
            st.session_state["threshold"] = threshold

    # Show results if already predicted
    if "prediction" in st.session_state:
        st.metric("Prediction", st.session_state["prediction"])
        st.metric("Risk Probability (Bad)", f"{st.session_state['proba']:.2f}")
        st.caption(f"⚖️ Decision Threshold = {st.session_state['threshold']:.2f}")

# ------------------------
# Explain Tab
# ------------------------
with tab2:
    st.subheader("📊 Explain Prediction (SHAP)")

    if "form_data" in st.session_state:
        if st.button("🔍 Explain Risk"):
            try:
                df = pd.DataFrame([st.session_state["form_data"]])
                preprocessed = model.named_steps["preprocessor"].transform(df)

                explainer = shap.TreeExplainer(model.named_steps["classifier"])
                shap_values = explainer(preprocessed)

                # ---- Show last prediction info with threshold ----
                st.markdown("### 🔮 Last Prediction")
                st.write(f"**Prediction:** {st.session_state['prediction']}")
                st.write(f"**Risk Probability (Bad):** {st.session_state['proba']:.2f}")
                st.caption(f"⚖️ Decision Threshold Used = {st.session_state['threshold']:.2f}")

                # ---- Plot SHAP summary ----
                st.write("### SHAP Feature Impact")
                fig, ax = plt.subplots()
                shap.plots.bar(shap_values[0], show=False)
                st.pyplot(fig)

                # ---- Top 5 Features Table ----
                values = shap_values.values[0]
                feature_names = model.named_steps["preprocessor"].get_feature_names_out()

                feature_importance = pd.DataFrame({
                    "Feature": [feature_name_map.get(f, f) for f in feature_names],
                    "SHAP Value": values
                }).sort_values(
                    by="SHAP Value", key=abs, ascending=False
                ).head(5)

                def color_shap(val):
                    color = "green" if val < 0 else "red"
                    return f"color: {color}; font-weight: bold"

                st.write("### 📋 Top 5 Contributing Features")
                st.dataframe(
                    feature_importance.style.format({"SHAP Value": "{:.3f}"}).applymap(color_shap, subset=["SHAP Value"])
                )

                st.markdown("""
                **Legend**:  
                🟢 Negative SHAP → Supports Good Credit  
                🔴 Positive SHAP → Supports Bad Credit  
                """)

            except Exception as e:
                st.error(f"Explainability failed: {e}")
    else:
        st.info("⚠️ Please make a prediction first before running SHAP explainability.")

# ✅ Clarification Note for Users
st.info("""
ℹ️ **Note:** The SHAP explanation is based on the model's raw probability output.  
Changing the decision threshold (e.g., from 0.5 → 0.35) does **not** change the SHAP values,  
it only changes how we classify the same probability as *Good* or *Bad Credit*.
""")    
# -----------------------------
# ------------------------
# Threshold Simulator Tab
# ------------------------
tab3 = st.tabs(["🧪 Threshold Simulator"])[0]

with tab3:
    st.subheader("🧪 Threshold Simulator")

    st.markdown("""
    This tool shows how the **decision flips** based on different thresholds for the same applicant risk probability.
    """)

    # Example risk probability (simulate an applicant score)
    prob = st.slider("📊 Simulated Applicant Risk Probability (Bad Credit)", 0.0, 1.0, 0.45, 0.01)

    # Choose thresholds to compare
    thresholds = [0.2, 0.3, 0.4, 0.5]

    results = []
    for th in thresholds:
        prediction = "❌ Bad Credit" if prob >= th else "✅ Good Credit"
        results.append({"Threshold": th, "Risk Probability": prob, "Prediction": prediction})

    df_results = pd.DataFrame(results)

    st.write("### Comparison Table")
    st.dataframe(df_results)

    # Plot for better visualization
    fig, ax = plt.subplots()
    ax.axhline(prob, color="blue", linestyle="--", label=f"Risk Probability = {prob:.2f}")

    for th in thresholds:
        ax.axvline(th, linestyle="--", alpha=0.6, label=f"Threshold {th:.2f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Probability")
    ax.set_title("Threshold vs. Applicant Risk")
    ax.legend()
    st.pyplot(fig)       