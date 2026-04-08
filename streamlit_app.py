# streamlit_app.py
# ==========================================================
# OpacGuard — Streamlit UI (complete, matches final 20-feature dataset)
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap
import ast
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from load_data import load_german_credit, columns as ORIGINAL_COLUMNS
from model_artifact import load_model_artifact, find_pipeline_steps, get_column_transformer, compute_engineered_features
from recommendation_engine.engine import RecommendationEngine
import json
from datetime import datetime


# -------------------------
# Utility helpers
# -------------------------

def clean_feat_names(names):
    """Strip typical prefixes like 'num__' or 'cat__' or 'preprocessor__' produced by get_feature_names_out."""
    cleaned = []
    for n in names:
        if "__" in n:
            cleaned.append(n.split("__")[-1])
        else:
            cleaned.append(n)
    return list(cleaned)

def sanitize(fn):
    return fn.replace(":", "").replace("/", "_").replace(" ", "_").replace("–", "-")

def create_fico_gauge(score):
    """Creates a beautiful Plotly multi-colored FICO gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Estimated FICO Score", 'font': {'size': 20, 'color': '#5d6d7e'}},
        number={'font': {'size': 60, 'color': '#707b7c'}},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#17202a"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "gray",
            'steps': [
                {'range': [300, 580], 'color': '#fdedec'}, # Poor
                {'range': [580, 670], 'color': '#fdf2e9'}, # Fair
                {'range': [670, 740], 'color': '#e8f8f5'}, # Good
                {'range': [740, 850], 'color': '#d5f5e3'}  # Excellent
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, b=20, t=50))
    return fig

def create_threshold_bullet_chart(applicant_proba, global_threshold):
    """Creates a premium Plotly Bullet Chart to visualize Applicant Risk vs Enterprise Threshold."""
    fig = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        value = applicant_proba,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'valueformat': '.3f'},
        delta = {'reference': global_threshold, 'position': "top", 'valueformat': '.3f'},
        title = {'text': "Applicant Risk<br>vs Policy", 'font': {"size": 18}},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [None, 1.0]},
            'threshold': {
                'line': {'color': "black", 'width': 5},
                'thickness': 0.75,
                'value': global_threshold
            },
            'steps': [
                {'range': [0.0, global_threshold], 'color': "#d5f5e3"}, # Safe Approval Zone
                {'range': [global_threshold, 1.0], 'color': "#fdedec"}  # High Risk Rejection Zone
            ],
            'bar': {'color': "gray"}
        }
    ))
    fig.update_layout(height=180, margin=dict(l=150, r=20, b=20, t=30))
    return fig

# -------------------------
# Constants
# -------------------------
MODEL_PATHS = [
    "credit_model.pkl"
]
RESULTS_DIR = "results"
SHAP_CSV = os.path.join(RESULTS_DIR, "shap_mean_abs_importance.csv")
OOF_PATH = os.path.join(RESULTS_DIR, "oof_preds.csv")


def load_friendly_map():
    """Load mapping of transformed feature name -> friendly label.
    Prefer the CSV produced by backend/shap_avg.py; otherwise parse the Python file for code_map.
    """
    # first try CSV
    if os.path.exists(SHAP_CSV):
        try:
            df = pd.read_csv(SHAP_CSV)
            if "feature_raw" in df.columns and "feature_label" in df.columns:
                return dict(zip(df["feature_raw"].astype(str), df["feature_label"].astype(str)))
        except Exception:
            pass

    # fallback: parse backend/shap_avg.py for `code_map` literal
    shap_avg_path = os.path.join("backend", "shap_avg.py")
    if os.path.exists(shap_avg_path):
        try:
            txt = open(shap_avg_path, "r", encoding="utf-8").read()
            idx = txt.find("code_map =")
            if idx != -1:
                sub = txt[idx:]
                # find the opening brace
                bidx = sub.find("{")
                if bidx != -1:
                    # balance braces
                    i = bidx
                    depth = 0
                    while i < len(sub):
                        if sub[i] == "{":
                            depth += 1
                        elif sub[i] == "}":
                            depth -= 1
                            if depth == 0:
                                dict_txt = sub[bidx:i+1]
                                try:
                                    return ast.literal_eval(dict_txt)
                                except Exception:
                                    break
                        i += 1
        except Exception:
            pass

    return {}

# Load friendly mapping once
FRIENDLY_MAP = load_friendly_map()

# simple JSONL interaction logger (append-only)
LOG_PATH = os.path.join(RESULTS_DIR, "interaction_log.jsonl")
def log_event(event_type, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {"time": datetime.utcnow().isoformat() + "Z", "event": event_type, "payload": payload}
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except Exception:
        pass


# -------------------------
# Engineered feature helper moved to model_artifact.py
# -------------------------


def _overlay_applicant_annotation(ax, x_app, y_app, df_inp, sel_feature, sel_interaction, reverse_map):
    """Draw a black X at (x_app, y_app) and annotate with applicant info (friendly mapping for A-codes).
    Annotation is placed in axes-fraction coords to avoid overlapping the colorbar when possible.
    """
    try:
        ax.scatter([x_app], [y_app], c="black", s=140, marker="X", edgecolors="white", linewidths=1.5)
    except Exception:
        try:
            ax.scatter([x_app], [y_app], c="black", s=60, marker="x")
        except Exception:
            return

    try:
        raw_display = None
        if df_inp is not None:
            if sel_feature in df_inp.columns:
                raw_display = df_inp.iloc[0].get(sel_feature)
            elif sel_interaction in df_inp.columns:
                raw_display = df_inp.iloc[0].get(sel_interaction)

        if isinstance(raw_display, str) and raw_display.startswith('A'):
            raw_disp_pretty = reverse_map.get(raw_display, raw_display)
        else:
            try:
                raw_disp_pretty = f"{float(raw_display):.3f}"
            except Exception:
                raw_disp_pretty = str(raw_display)

        interaction_val = None
        if df_inp is not None and sel_interaction != "auto" and sel_interaction in df_inp.columns:
            interaction_val = df_inp.iloc[0].get(sel_interaction)

        ann_lines = [f"Applicant: {raw_disp_pretty}", f"SHAP={y_app:.3f}"]
        if interaction_val is not None:
            ann_lines.append(f"{sel_interaction}: {interaction_val}")
        ann_text = "\n".join(ann_lines)

        # place annotation in axes-fraction space (avoids overlapping colorbar)
        try:
            disp = ax.transData.transform((x_app, y_app))
            fx, fy = ax.transAxes.inverted().transform(disp)
            fy = min(max(fy, 0.08), 0.92)
            if fx > 0.7:
                text_pos = (0.02, fy)
                ha = 'left'
            else:
                text_pos = (0.95, fy)
                ha = 'right'
            ax.annotate(ann_text, xy=(x_app, y_app), xycoords='data', xytext=text_pos, textcoords='axes fraction', fontsize=9, ha=ha,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.95),
                        arrowprops=dict(arrowstyle='->', lw=0.7))
        except Exception:
            try:
                ax.annotate(ann_text, xy=(x_app, y_app), xytext=(x_app + 0.03, y_app), fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.95))
            except Exception:
                pass
    except Exception:
        pass

# -------------------------
# 0. Page config
# -------------------------
st.set_page_config(page_title="OpacGuard — Credit Risk", layout="wide")
st.title("💳 OpacGuard — Credit Risk Prediction Dashboard")



# -------------------------
# 1. Load saved pipeline
# -------------------------
@st.cache_resource
def load_pipeline():
    return load_model_artifact()

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to load model pipeline: {e}")
    st.stop()

# -------------------------
# 1b. Load DiCE Explainer cached
# -------------------------
class CustomWrapper:
    def __init__(self, pl):
        self.pl = pl
    def predict_proba(self, X):
        X_eng = compute_engineered_features(X.copy())
        if hasattr(self.pl, "predict_proba"):
            return self.pl.predict_proba(X_eng)
        pre, clf, _ = find_pipeline_steps(self.pl)
        return clf.predict_proba(pre.transform(X_eng))
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

@st.cache_resource
def get_dice_explainer():
    import dice_ml
    df = load_german_credit()
    
    used_cols = [
        "Status_of_existing_checking_account", "Duration_in_month", "Credit_history", "Purpose", "Credit_amount",
        "Savings_account/bonds", "Present_employment_since", "Personal_status_and_sex", "Other_debtors/guarantors",
        "Property", "Age_in_years", "Other_installment_plans", "Housing", "Job", "target"
    ]
    df_used = df[used_cols]
    
    continuous_features = ['Duration_in_month', 'Credit_amount', 'Age_in_years']
    w = CustomWrapper(pipeline)
    d = dice_ml.Data(dataframe=df_used, continuous_features=continuous_features, outcome_name='target')
    m = dice_ml.Model(model=w, backend='sklearn')
    return dice_ml.Dice(d, m, method="random")

preprocessor, classifier, pipeline_ref = find_pipeline_steps(pipeline)

# -------------------------
# 2. Categorical code mappings (friendly -> raw A-codes)
# -------------------------
# These produce the exact raw values your pipeline expects (A11, A12, ...).
Status_map = {"Checking <0 DM": "A11", "Checking 0–200 DM": "A12", "Checking ≥200 DM": "A13", "No Checking Account": "A14"}

CreditHistory_map = {
    "No credit taken": "A30",
    "All paid duly": "A31",
    "Existing paid (delay)": "A32",
    "Existing paid (other)": "A33",
    "Critical / other loans": "A34"
}

Purpose_map = {
    "New Car": "A40", "Used Car": "A41", "Furniture": "A42", "Radio/TV": "A43",
    "Appliances": "A44", "Repairs": "A45", "Education": "A46", "Vacation": "A47",
    "Retraining": "A48", "Business": "A49", "Other": "A410"
}

Savings_map = {
    "Savings <100 DM": "A61", "Savings 100–500 DM": "A62", "Savings 500–1000 DM": "A63",
    "Savings ≥1000 DM": "A64", "No/Unknown": "A65"
}

Employment_map = {"Unemployed": "A71", "<1 yr": "A72", "1–4 yrs": "A73", "4–7 yrs": "A74", "≥7 yrs": "A75"}

PersonalStatus_map = {
    "Male: Div/Separated": "A91", "Female: Single": "A92", "Male: Single": "A93",
    "Male: Married/Widowed": "A94", "Female: Div/Married": "A95"
}

OtherDebtors_map = {"None": "A101", "Co-applicant": "A102", "Guarantor": "A103"}

Property_map = {"Real estate": "A121", "Savings/Insurance": "A122", "Car/Other": "A123", "None": "A124"}

OtherInstallments_map = {"Bank": "A141", "Stores": "A142", "None": "A143"}

Housing_map = {"Rent": "A151", "Own": "A152", "Free": "A153"}

Job_map = {"Unskilled NonResident": "A171", "Unskilled Resident": "A172", "Skilled": "A173", "Highly Skilled": "A174"}

# -------------------------
# 3. Layout: Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predict Credit Risk", "📊 Explain Prediction", "🧪 Threshold Simulator", "💡 Actionable Recommendations", "🔐 MLOps Dashboard (Admin)"])

# -------------------------
# 4. Predict Tab: user inputs for original columns (engineered features will be computed)
# -------------------------
with tab1:
    st.header("📋 Enter Applicant Information (original features only)")
    st.markdown("The pipeline will compute engineered features (ratios, logs, age bin) automatically from your inputs.")

    # threshold sidebar
    threshold = st.sidebar.slider("Decision threshold (Bad if probability ≥ threshold)", 0.1, 0.9, 0.5, 0.05)

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            Status_choice = st.selectbox("Status of existing checking account", list(Status_map.keys()))
            Duration_in_month = st.number_input("Duration (months)", min_value=4, max_value=72, value=12)
            Credit_amount = st.number_input("Credit amount", min_value=250, max_value=20000, value=1000, step=50)
            Savings_choice = st.selectbox("Savings account / bonds", list(Savings_map.keys()))
            Present_employment_since = st.selectbox("Present employment since", list(Employment_map.keys()))
            Personal_status_and_sex = st.selectbox("Personal status & sex", list(PersonalStatus_map.keys()))

        with col2:
            Credit_history_choice = st.selectbox("Credit history", list(CreditHistory_map.keys()))
            Purpose_choice = st.selectbox("Purpose", list(Purpose_map.keys()))
            Other_debtors_choice = st.selectbox("Other debtors / guarantors", list(OtherDebtors_map.keys()))
            Property_choice = st.selectbox("Property", list(Property_map.keys()))
            Age_in_years = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
            Other_installment_plans_choice = st.selectbox("Other installment plans", list(OtherInstallments_map.keys()))
            Housing_choice = st.selectbox("Housing", list(Housing_map.keys()))
            Job_choice = st.selectbox("Job", list(Job_map.keys()))

        submitted = st.form_submit_button("🔮 Predict")

        if submitted:
           try:
                
              # --------------------------------------------------
                # 1. Build RAW input (German dataset schema)
                # --------------------------------------------------
                df_raw = pd.DataFrame([{
                    "Status_of_existing_checking_account": Status_map[Status_choice],
                    "Duration_in_month": Duration_in_month,
                    "Credit_history": CreditHistory_map[Credit_history_choice],
                    "Purpose": Purpose_map[Purpose_choice],
                    "Credit_amount": Credit_amount,
                    "Savings_account/bonds": Savings_map[Savings_choice],
                    "Present_employment_since": Employment_map[Present_employment_since],
                    "Personal_status_and_sex": PersonalStatus_map[Personal_status_and_sex],
                    "Other_debtors/guarantors": OtherDebtors_map[Other_debtors_choice],
                    "Property": Property_map[Property_choice],
                    "Age_in_years": Age_in_years,
                    "Other_installment_plans": OtherInstallments_map[Other_installment_plans_choice],
                    "Housing": Housing_map[Housing_choice],
                    "Job": Job_map[Job_choice],
                }])

            # --------------------------------------------------
            # 2. Compute ENGINEERED FEATURES (AS TRAINED)
            # --------------------------------------------------
                df_eng = compute_engineered_features(df_raw.copy())

            # --------------------------------------------------
            # 3. Predict on ENGINEERED DATA
            # --------------------------------------------------
                if hasattr(pipeline_ref, "predict_proba"):
                    proba = pipeline_ref.predict_proba(df_eng)[0][1]
                else:
                    if preprocessor is None or classifier is None:
                        raise RuntimeError("Preprocessor or classifier not found inside pipeline object.")
                    Xt = preprocessor.transform(df_eng)
                    proba = classifier.predict_proba(Xt)[0][1]

                label = "Bad Credit ❌" if proba >= threshold else "Good Credit ✅"

                st.session_state["proba"] = proba
                st.session_state["pred_label"] = label

            # --------------------------------------------------
            # 4. Save for Explain tab
            # --------------------------------------------------
                st.session_state["input_df"] = df_eng
                st.session_state["input_df_raw"] = df_raw

                # UI is rendered dynamically below form
                st.session_state["predict_clicked"] = True
                
           except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Show available prediction (if any)
    if "pred_label" in st.session_state and st.session_state.get("predict_clicked", False):
        pred_label = st.session_state["pred_label"]
        proba = st.session_state["proba"]
        fico_score = int(300 + (1.0 - proba) * 550)
        
        # Determine Risk Tier based on FICO
        if fico_score < 580:
            tier_color, tier_text = "red", "Poor"
            badge_color = "#fdedec"
            text_color = "red"
        elif fico_score < 670:
            tier_color, tier_text = "orange", "Fair"
            badge_color = "#fdf2e9"
            text_color = "orange"
        elif fico_score < 740:
            tier_color, tier_text = "green", "Good"
            badge_color = "#e8f8f5"
            text_color = "green"
        else:
            tier_color, tier_text = "darkgreen", "Excellent"
            badge_color = "#d5f5e3"
            text_color = "darkgreen"
            
        st.markdown("---")
        # Top KPI row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(f"Model label ({threshold:.2f})", pred_label)
        kpi2.metric("Adjusted label", pred_label)
        kpi3.metric("Risk probability (Bad)", f"{proba:.3f}")
        
        with kpi4:
            st.metric("Credit Score", fico_score)
            st.markdown(f"<span style='background-color:{badge_color}; color:{text_color}; padding:3px 8px; border-radius:4px; font-weight:bold; font-size:12px;'>↑ {tier_text}</span>", unsafe_allow_html=True)
            
        st.caption(f"Decision threshold used: {threshold:.2f}")
        tier_label = "High" if proba >= 0.50 else "Low"
        tier_bg = "#fdedec" if tier_label == "High" else "#e8f8f5"
        tier_fc = "red" if tier_label == "High" else "green"
        st.markdown(f"<span style='background-color:{tier_bg}; color:{tier_fc}; padding:6px 12px; border-radius:6px; font-weight:bold;'>Risk Tier: {tier_label}</span>", unsafe_allow_html=True)
        
        # Inject the Gauge Chart
        st.plotly_chart(create_fico_gauge(fico_score), use_container_width=True)



# -------------------------
# 5. Explain Tab (SHAP + top-5 table + dependence)
# -------------------------
with tab2:
    st.header("📊 Explain Prediction (SHAP)")

    if "input_df" not in st.session_state:
        st.info("Make a prediction first under the Predict tab (top-left).")
    else:
        # compute or re-use existing SHAP results stored in session_state
        if st.button("🔍 Generate SHAP explanation for last input") or "shap_cached" not in st.session_state:
            df_row = st.session_state["input_df"]

            # Check we have preprocessor & classifier available
            if preprocessor is None or classifier is None:
                st.error("Pipeline does not expose preprocessor and classifier steps. SHAP cannot be generated.")
            else:
                try:
                    # transform -> try to produce a DataFrame with proper column names
                    X_trans = preprocessor.transform(df_row)

                    # attempt to get transformed feature names
                    feat_names_raw = None
                    try:
                        if hasattr(preprocessor, "get_feature_names_out"):
                            feat_names_raw = preprocessor.get_feature_names_out()
                        elif hasattr(preprocessor, "named_steps"):
                            for step in preprocessor.named_steps.values():
                                if hasattr(step, "get_feature_names_out"):
                                    feat_names_raw = step.get_feature_names_out()
                                    break
                        if feat_names_raw is None and hasattr(preprocessor, "transformers_"):
                            feat_names_raw = preprocessor.get_feature_names_out()
                    except Exception:
                        feat_names_raw = None

                    if feat_names_raw is None:
                        feat_names = [f"f{i}" for i in range(X_trans.shape[1])]
                    else:
                        feat_names = clean_feat_names(list(feat_names_raw))

                    # If feat_names look generic (f0,f1,...) and we have a SHAP CSV with the true
                    # transformed feature order, prefer that order so we can map to friendly labels
                    try:
                        if all(str(fn).startswith("f") for fn in feat_names) and os.path.exists(SHAP_CSV):
                            df_shap_map = pd.read_csv(SHAP_CSV)
                            if "feature_raw" in df_shap_map.columns:
                                full_list = df_shap_map["feature_raw"].astype(str).tolist()
                                # if lengths match, adopt the CSV ordering; if CSV longer, take prefix
                                if len(full_list) >= X_trans.shape[1]:
                                    feat_names = full_list[: X_trans.shape[1]]
                    except Exception:
                        pass

                    # Ensure X_trans is 2D numpy
                    X_arr = X_trans if isinstance(X_trans, (np.ndarray,)) else np.asarray(X_trans)

                    # Prepare a small background sample (training data transformed) for dependence plots
                    try:
                        df_train = load_german_credit()
                        df_train_eng = compute_engineered_features(df_train)
                        n_bg = min(200, len(df_train_eng))
                        X_bg = preprocessor.transform(df_train_eng.head(n_bg))
                        X_bg = X_bg if isinstance(X_bg, np.ndarray) else np.asarray(X_bg)
                    except Exception:
                        X_bg = None

                    # SHAP explainer — prefer TreeExplainer for tree models
                    try:
                        explainer = shap.TreeExplainer(classifier)
                        shap_expl_single = explainer(X_arr)
                        shap_vals_single = shap_expl_single.values
                        shap_expl_bg = explainer(X_bg) if X_bg is not None else None
                    except Exception:
                        explainer = shap.Explainer(classifier, X_arr)
                        shap_expl_single = explainer(X_arr)
                        shap_vals_single = shap_expl_single.values
                        shap_expl_bg = explainer(X_bg) if X_bg is not None else None

                    # store in session for interactivity without recompute
                    st.session_state["shap_cached"] = True
                    st.session_state["shap_vals_single"] = shap_vals_single
                    st.session_state["shap_expl_single"] = shap_expl_single
                    st.session_state["X_trans"] = X_arr
                    st.session_state["feat_names"] = feat_names
                    st.session_state["df_input_raw"] = df_row.copy()
                    st.session_state["X_bg"] = X_bg
                    st.session_state["shap_expl_bg"] = shap_expl_bg
                    # log the event
                    try:
                        log_event("shap_generated", {"feat_count": len(feat_names), "has_bg": X_bg is not None})
                    except Exception:
                        pass

                except Exception as e:
                    st.error(f"SHAP explain failed: {e}")

        # if cached, render the visuals
        if "shap_cached" in st.session_state and st.session_state.get("shap_vals_single") is not None:
            shap_vals = st.session_state["shap_vals_single"]
            X_arr = st.session_state["X_trans"]
            feat_names = st.session_state["feat_names"]
            df_row = st.session_state["df_input_raw"]
            shap_expl_single = st.session_state.get("shap_expl_single")
            shap_expl_bg = st.session_state.get("shap_expl_bg")
            X_bg = st.session_state.get("X_bg")

            # Build reverse mapping for categorical codes to friendly labels
            reverse_map = {}
            for m in [Status_map, CreditHistory_map, Purpose_map, Savings_map, Employment_map, PersonalStatus_map,
                      OtherDebtors_map, Property_map, OtherInstallments_map, Housing_map, Job_map]:
                for k, v in m.items():
                    reverse_map[v] = k

            # Use FRIENDLY_MAP when available to prettify transformed feature names
            def prettify_raw(fn):
                return FRIENDLY_MAP.get(fn, fn)

            # helper: get friendly label for a transformed feature name
            def get_friendly_and_value(feat_name):
                # Normalize name (strip any prefix before last '__')
                norm = feat_name.split("__")[-1]

                # Special handling for age_bin one-hot style features: e.g. 'age_bin_26-35'
                if norm.startswith("age_bin"):
                    # extract bin code after the first underscore (if present)
                    parts = norm.split("_", 1)
                    code = parts[1] if len(parts) > 1 else None
                    friendly = "Age Bin" if code is None else f"Age Bin: {code}"
                    # the applicant's engineered age_bin value (from compute_engineered_features)
                    val = df_row.iloc[0].get("age_bin", None)
                    # raw value is the bin (e.g., '26-35'); input_val indicates whether this one-hot column is active
                    raw_val = val
                    input_val = 1 if (code is not None and val == code) else 0 if code is not None else val
                    applicant_label = val
                    return friendly, raw_val, input_val, applicant_label

                # one-hot style: try to detect trailing code like '_A11' or 'A11'
                if "_A" in norm or norm.endswith(tuple(["A11","A12","A13","A14","A30","A31","A32","A33","A34","A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A61","A62","A63","A64","A65"])):
                    # Try rsplit on '_' to separate base and code
                    parts = norm.rsplit("_", 1)
                    if len(parts) == 2:
                        base, code = parts[0], parts[1]
                    else:
                        # fallback if no underscore but endswith code
                        # attempt to find code at end (last 3 chars)
                        base = norm[:-3]
                        code = norm[-3:]
                    friendly = reverse_map.get(code, base)
                    input_val = None
                    # Try several strategies to find the original column storing the code
                    candidates = [base, base.replace("-", "_").replace("/", "_")]
                    candidates += [c for c in df_row.columns if base in c]
                    found = False
                    for cand in candidates:
                        if cand in df_row.columns:
                            input_code = df_row.iloc[0].get(cand)
                            if input_code == code:
                                input_val = reverse_map.get(code, code)
                            else:
                                input_val = "0"
                            found = True
                            break
                    if not found:
                        # last resort: check any column equal to code
                        for col in df_row.columns:
                            if df_row.iloc[0].get(col) == code:
                                input_val = reverse_map.get(code, code)
                                found = True
                                break
                    # Final fallback: if still None, try friendly map for this transformed feature
                    if input_val is None:
                        # If FRIENDLY_MAP contains an entry for this exact transformed name, use it
                        fm = FRIENDLY_MAP.get(norm) or FRIENDLY_MAP.get(feat_name)
                        if fm:
                            input_val = fm
                        else:
                            # try mapping base_code -> friendly
                            candidate = f"{base}_{code}"
                            input_val = FRIENDLY_MAP.get(candidate) or reverse_map.get(code) or code
                    # Also compute the applicant's selected label for the base column (if available)
                    applicant_label = None
                    for cand in candidates:
                        if cand in df_row.columns:
                            code = df_row.iloc[0].get(cand)
                            applicant_label = reverse_map.get(code)
                            break
                    # final fallback try exact base column name
                    if applicant_label is None and base in df_row.columns:
                        applicant_label = reverse_map.get(df_row.iloc[0].get(base))
                    # raw code value (e.g., 'A14') when available
                    raw_val = None
                    try:
                        for cand in candidates:
                            if cand in df_row.columns:
                                raw_val = df_row.iloc[0].get(cand)
                                break
                    except Exception:
                        raw_val = None
                    return friendly, raw_val, input_val, applicant_label

                # engineered / numeric
                if norm in ["credit_per_month", "credit_to_age", "log_credit_amount", "log_duration_in_month", "age_bin"] or norm in df_row.columns:
                    label_map = {
                        "credit_per_month": "Credit per Month",
                        "credit_to_age": "Credit-to-Age Ratio",
                        "log_credit_amount": "Log(Credit Amount)",
                        "log_duration_in_month": "Log(Duration Months)",
                        "age_bin": "Age Bin",
                    }
                    val = df_row.iloc[0].get(norm, None)
                    # For engineered/numeric, raw_val is same as val; applicant_label not applicable
                    return label_map.get(norm, norm), val, val, None

                # fallback: return raw name
                return norm, None, None, None

            # Show a waterfall plot for the single instance (clean and focused)
            st.subheader("Feature contributions (waterfall)")
            try:
                fig_w = plt.figure(figsize=(8, 4))
                # shap expects shap_values for the instance; new API uses Explanation objects but we have array
                single_shap = np.array(shap_vals)[0]
                # create a pandas Series indexed by feat_names and sort by magnitude for display in waterfall
                # Use shap.plots.waterfall requires an Explanation or value dict; use shap.plots.bar for fallback
                try:
                    # Prefer to build a fresh SHAP Explanation with friendly feature names and
                    # display-data aligned to what we show in the Top-5 table so the waterfall
                    # left-hand values match the table's Value (friendly).
                    try:
                        pretty_feature_names = [FRIENDLY_MAP.get(fn, prettify_raw(fn)) for fn in feat_names]

                        # Build display data for this instance using the same helper we used for the table.
                        # For clarity, include a combined string: '<transformed_value> | <friendly_value>' when possible.
                        display_data = []
                        for fn in feat_names:
                            # get_friendly_and_value returns (friendly_label, raw_val, input_val, applicant_label)
                            try:
                                lab, rawv, val, appl = get_friendly_and_value(fn)
                            except Exception:
                                lab, rawv, val, appl = (FRIENDLY_MAP.get(fn, prettify_raw(fn)), None, None, None)
                            # Build a combined display entry that shows both the underlying transformed value
                            # (rawv) and the friendly label/value (val). Use rawv first, then friendly val.
                            try:
                                transformed_display = rawv if rawv is not None else ''
                                friendly_display = val if val is not None else ''
                                if str(transformed_display) and str(friendly_display) and str(transformed_display) != str(friendly_display):
                                    combined = f"{transformed_display} | {friendly_display}"
                                else:
                                    combined = str(transformed_display or friendly_display)
                            except Exception:
                                combined = str(rawv or val or '')
                            display_data.append(combined)

                        # Ensure arrays are correct shapes
                        vals = np.array(shap_vals)
                        # base value (expected) — try common attribute names
                        basev = None
                        try:
                            basev = getattr(shap_expl_single, 'base_values', None)
                        except Exception:
                            basev = None
                        if basev is None:
                            try:
                                basev = getattr(shap_expl_single, 'expected_value', None)
                            except Exception:
                                basev = None


                        # Build a new Explanation object if shap supports it; fallback will catch problems
                        try:
                            new_expl = shap.Explanation(values=vals, base_values=basev, data=np.atleast_2d(display_data), feature_names=pretty_feature_names)
                            # shap expects Explanation with shape (n_samples, n_features) for values
                            # If values is (1,nf) this should work; call waterfall on the first sample
                            shap.plots.waterfall(new_expl[0], show=False)
                            st.pyplot(fig_w)
                        except Exception:
                            # Some shap versions require different constructor shapes; try alternative
                            try:
                                new_expl = shap.Explanation(values=vals[0], base_values=(basev[0] if isinstance(basev, (list, np.ndarray)) else basev), data=np.atleast_2d(display_data), feature_names=pretty_feature_names)
                                shap.plots.waterfall(new_expl, show=False)
                                st.pyplot(fig_w)
                            except Exception:
                                # let outer except handle fallback bar
                                raise
                    except Exception:
                        # If any of the above fails, fall through to the fallback bar below
                        raise
                except Exception:
                    # fallback simple bar of top features
                    df_loc = pd.DataFrame({"feature": feat_names, "shap": single_shap})
                    # replace raw feature names with FRIENDLY_MAP where available
                    df_loc["feature_label"] = df_loc["feature"].apply(lambda f: FRIENDLY_MAP.get(f, prettify_raw(f)))
                    df_loc = df_loc.set_index("feature_label").iloc[:15]
                    df_loc["abs"] = df_loc["shap"].abs()
                    df_loc = df_loc.sort_values("abs", ascending=True)
                    colors = df_loc["shap"].apply(lambda v: "red" if v>0 else "green")
                    df_loc["shap"].plot.barh(figsize=(8, 4), color=colors)
                    plt.xlabel("SHAP value")
                    st.pyplot(plt.gcf())
                plt.close("all")
            except Exception as e:
                st.warning(f"Couldn’t render waterfall/bar plot: {e}")

            # Top contributing features table with friendly names and actual input values
            st.subheader("Top contributing features for this applicant")
            row_shap = np.array(shap_vals)[0]
            df_shap = pd.DataFrame({"feature_raw": feat_names, "shap_value": row_shap})
            df_shap["abs_shap"] = df_shap["shap_value"].abs()
            top_shap = df_shap.sort_values("abs_shap", ascending=False).head(15).copy()

            # Build display columns (capture raw + friendly values)
            labels = []
            raw_vals = []
            input_vals = []
            applicant_labels = []
            for fn in top_shap["feature_raw"]:
                lab, rawv, val, appl = get_friendly_and_value(fn)
                labels.append(lab)
                raw_vals.append(rawv)
                input_vals.append(val)
                applicant_labels.append(appl)


            # Try to use FRIENDLY_MAP for the transformed raw feature name; fall back to our label detection
            pretty_labels = []
            for raw, lab in zip(top_shap["feature_raw"], labels):
                pretty = FRIENDLY_MAP.get(raw)
                if pretty is None:
                    pretty = lab
                pretty_labels.append(pretty)

            top_shap["feature_label"] = pretty_labels
            top_shap["value_raw"] = raw_vals
            top_shap["value_friendly"] = input_vals
            display_df = top_shap[["feature_label", "value_raw", "value_friendly", "shap_value"]].rename(columns={"feature_label": "Feature", "value_raw": "Value (raw)", "value_friendly": "Value (friendly)", "shap_value": "SHAP value"})
            # highlight top-5 with cards
            st.markdown("**Top 5 features (highest |SHAP|)**")
            top5 = top_shap.head(5).copy()
            cols = st.columns(5)
            for i, (_, row) in enumerate(top5.iterrows()):
                with cols[i]:
                    fn = row["feature_label"]
                    rawv = row.get("value_raw") if pd.notna(row.get("value_raw")) else ""
                    val = row.get("value_friendly") if pd.notna(row.get("value_friendly")) else ""
                    appl = applicant_labels[i] if i < len(applicant_labels) else None
                    shapv = row["shap_value"]
                    color = "#ffdddd" if shapv > 0 else "#ddffdd"
                    appl_line = f"Applicant: **{appl}**  \n\n" if appl else ""
                    st.markdown(f"<div style='background:{color}; padding:10px; border-radius:6px;'>\n**{fn}**  \n\n{appl_line}Value (raw): `{rawv}`  \n\nValue (friendly): `{val}`  \n\nSHAP: `{shapv:.4f}`\n</div>", unsafe_allow_html=True)

            # show table for remaining
            st.dataframe(display_df.style.format({"SHAP value": "{:.4f}"}).applymap(lambda v: "color: red" if v > 0 else "color: green", subset=["SHAP value"]))

            # Dependence plot selector: allow picking from stored feat_names
            st.subheader("Dependence plot (choose a feature)")

            def _pretty_select_label(fn):
                # show both the transformed raw name and friendly label (and code for A-codes)
                try:
                    if fn == "auto":
                        return "auto"
                    # detect trailing A-code like '_A14' or ending with 'A14'
                    if "_A" in fn or fn.endswith(tuple(["A11","A12","A13","A14","A30","A31","A32","A33","A34","A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A61","A62","A63","A64","A65"])):
                        parts = fn.rsplit("_", 1)
                        if len(parts) == 2:
                            base, code = parts[0], parts[1]
                        else:
                            base = fn[:-3]
                            code = fn[-3:]
                        friendly = reverse_map.get(code, None)
                        if friendly is not None:
                            return f"{fn} ({code} — {friendly})"
                        return f"{fn} ({code})"
                    # otherwise use FRIENDLY_MAP when available
                    return FRIENDLY_MAP.get(fn, fn)
                except Exception:
                    return fn

            sel_feature = st.selectbox("Feature (raw transformed names)", feat_names, index=0, key="dependence_select", format_func=_pretty_select_label)
            # interaction selector: 'auto' or pick a second feature to color by
            interaction_options = ["auto"] + feat_names
            sel_interaction = st.selectbox("Interaction feature (color by)", interaction_options, index=0, key="dependence_interaction", format_func=_pretty_select_label)
            try:
                log_event("dependence_selected", {"feature": sel_feature, "interaction": sel_interaction})
            except Exception:
                pass

            try:
                # Convert arrays to DataFrame with columns=feat_names for clearer plotting and inspection
                Xbg_df = None
                Xarr_df = None
                if X_bg is not None and feat_names is not None:
                    try:
                        Xbg_df = pd.DataFrame(X_bg, columns=feat_names)
                    except Exception:
                        # fallback: create DataFrame with generic column names
                        Xbg_df = pd.DataFrame(X_bg)
                if X_arr is not None and feat_names is not None:
                    try:
                        Xarr_df = pd.DataFrame(X_arr, columns=feat_names)
                    except Exception:
                        Xarr_df = pd.DataFrame(X_arr)

                # Show a tiny debug table for the selected feature so the user can see dtype/unique values
                st.markdown("**Selected feature diagnostics**")
                try:
                    def _fmt_val(v):
                        if pd.isna(v):
                            return None
                        # map A-codes to friendly labels
                        if isinstance(v, str) and v.startswith('A'):
                            return reverse_map.get(v, v)
                        # convert numpy numeric scalars to native python types for cleaner repr
                        if isinstance(v, (np.floating, np.integer)):
                            try:
                                # if it's effectively an int, show as int
                                fv = float(v)
                                if abs(fv - int(fv)) < 1e-9:
                                    return int(fv)
                                return round(fv, 3)
                            except Exception:
                                return float(v)
                        return v

                    if Xbg_df is not None and sel_feature in Xbg_df.columns:
                        col_series = Xbg_df[sel_feature].head(10).reset_index(drop=True)
                        sample_display = [_fmt_val(v) for v in col_series.tolist()]
                        unique_vals = pd.unique(Xbg_df[sel_feature])[:10]
                        unique_display = [_fmt_val(v) for v in list(unique_vals)]
                        st.table(pd.DataFrame({"sample_values": sample_display}))
                        st.write(f"Unique (up to 10): {unique_display} | dtype: {Xbg_df[sel_feature].dtype}")
                    elif Xarr_df is not None and sel_feature in Xarr_df.columns:
                        col_series = Xarr_df[sel_feature].head(10).reset_index(drop=True)
                        sample_display = [_fmt_val(v) for v in col_series.tolist()]
                        unique_vals = pd.unique(Xarr_df[sel_feature])[:10]
                        unique_display = [_fmt_val(v) for v in list(unique_vals)]
                        st.table(pd.DataFrame({"sample_values": sample_display}))
                        st.write(f"Unique (up to 10): {unique_display} | dtype: {Xarr_df[sel_feature].dtype}")
                    else:
                        st.write("No background DataFrame available to show diagnostics for this feature.")
                except Exception as e:
                    st.write(f"Couldn’t compute diagnostics: {e}")

                fig2, ax = plt.subplots(figsize=(6, 4))
                # For dependence we need shap values across a sample — use background DataFrame if available
                # Prepare plotting DataFrame and handle one-hot / label-encoded mapping
                plot_df = None
                shap_vals_bg = None
                try:
                    # If we have background explanation and df, use it
                    if shap_expl_bg is not None and Xbg_df is not None:
                        shap_vals_bg = shap_expl_bg.values
                        plot_df = Xbg_df.copy()
                    elif Xarr_df is not None:
                        shap_vals_bg = shap_vals
                        plot_df = Xarr_df.copy()
                    else:
                        shap_vals_bg = shap_vals
                        plot_df = pd.DataFrame(X_arr, columns=feat_names) if feat_names is not None else pd.DataFrame(X_arr)

                    # If selected feature looks like a one-hot (endswith _Axx), derive binary series from original training data
                    bin_series = None
                    use_manual_plot = False
                    if "_A" in sel_feature or sel_feature.endswith(tuple(["A11","A12","A13","A14","A30","A31","A32","A33","A34","A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A61","A62","A63","A64","A65"])):
                        parts = sel_feature.rsplit("_", 1)
                        if len(parts) == 2:
                            base, code = parts[0], parts[1]
                        else:
                            base = sel_feature[:-3]
                            code = sel_feature[-3:]
                        # load raw train and compute engineered features to find base column values
                        try:
                            df_train = load_german_credit()
                            df_train_eng = compute_engineered_features(df_train)
                            # create binary series for plotting: 1 if base column equals code else 0
                            if base in df_train_eng.columns:
                                bin_series = (df_train_eng[base] == code).astype(int).reset_index(drop=True)
                                # adjust length to match plot_df
                                if len(bin_series) >= len(plot_df):
                                    bin_series = bin_series.head(len(plot_df))
                                else:
                                    if len(bin_series) > 0:
                                        repeats = int(np.ceil(len(plot_df) / len(bin_series)))
                                        bin_series = pd.concat([bin_series]*repeats, ignore_index=True).head(len(plot_df))
                                use_manual_plot = True
                        except Exception:
                            bin_series = None
                    plot_x_col = sel_feature if not use_manual_plot else None

                    # If interaction is a base categorical that contains A-codes, map codes to friendly labels for coloring
                    interaction_idx = None if sel_interaction == "auto" else sel_interaction
                    if interaction_idx is not None and plot_df is not None and interaction_idx in plot_df.columns:
                        # try convert numeric/encoded codes to friendly labels using reverse_map
                        try:
                            # If values look like strings 'A11' etc, map them
                            sample_vals = plot_df[interaction_idx].dropna().astype(str)
                            if sample_vals.str.startswith("A").any():
                                plot_df[interaction_idx] = plot_df[interaction_idx].astype(str).map(lambda v: reverse_map.get(v, v))
                        except Exception:
                            pass

                    # If we created a binary series, do manual plotting to avoid mismatch between shap values and features
                    if use_manual_plot and bin_series is not None:
                        try:
                            # determine the SHAP column index for the feature (if present)
                            if sel_feature in feat_names:
                                idx = feat_names.index(sel_feature)
                                y_vals = np.array(shap_vals_bg)[:, idx]
                            else:
                                y_vals = np.zeros(len(plot_df))

                            x_vals = bin_series.values

                            # prepare color values from interaction if requested
                            color_vals = None
                            color_labels = None
                            if interaction_idx is not None and interaction_idx in plot_df.columns:
                                try:
                                    col = plot_df[interaction_idx]
                                    if col.dtype.name == 'category' or col.dtype == object:
                                        codes, uniques = pd.factorize(col)
                                        color_vals = codes
                                        color_labels = list(uniques)
                                    else:
                                        color_vals = col.values
                                except Exception:
                                    color_vals = None

                            sc = ax.scatter(x_vals, y_vals, c=color_vals, cmap='plasma', s=40)
                            if color_vals is not None:
                                try:
                                    cbar = plt.colorbar(sc, ax=ax)
                                    # if discrete labels exist, set ticks and labels
                                    if color_labels is not None:
                                        cbar.set_ticks(np.arange(len(color_labels)) + 0.5)
                                        cbar.set_ticklabels(color_labels)
                                    # label colorbar with chosen interaction feature
                                    try:
                                        if sel_interaction != "auto":
                                            cbar.set_label(sel_interaction)
                                    except Exception:
                                        pass
                                except Exception:
                                    pass

                            # overlay applicant point using unified helper
                            try:
                                if 'shap_vals' in locals() and shap_vals is not None and sel_feature in feat_names:
                                    y_app = float(np.array(shap_vals)[0][idx])
                                    df_inp = st.session_state.get("input_df")
                                    x_app = None
                                    if df_inp is not None:
                                        base_val = df_inp.iloc[0].get(base)
                                        x_app = 1 if base_val == code else 0
                                    if x_app is not None:
                                        _overlay_applicant_annotation(ax, x_app, y_app, df_inp, sel_feature, sel_interaction, reverse_map)
                            except Exception:
                                pass

                            ax.set_xlabel(sel_feature)
                            ax.set_ylabel(f"SHAP value for {sel_feature}")
                        except Exception as e:
                            raise
                    else:
                        # call SHAP dependence plot with DataFrame so axes are labeled
                        fnames_for_shap = list(plot_df.columns) if plot_df is not None else feat_names
                        # Prepare DataFrame for SHAP and factorize discrete interaction columns so we can relabel ticks
                        try:
                            plot_df_for_shap = plot_df.copy() if plot_df is not None else None
                            interaction_to_pass = interaction_idx
                            friendly_labels_for_interaction = None

                            if interaction_idx is not None and plot_df_for_shap is not None and interaction_idx in plot_df_for_shap.columns:
                                try:
                                    col = plot_df_for_shap[interaction_idx]
                                    sample_vals = col.dropna().astype(str)
                                    looks_like_acode = sample_vals.str.startswith('A').any()
                                except Exception:
                                    looks_like_acode = False

                                try:
                                    # If the column appears categorical/discrete (A-codes or low cardinality), factorize for SHAP
                                    if looks_like_acode or (plot_df_for_shap[interaction_idx].dtype == object and plot_df_for_shap[interaction_idx].nunique() < 30):
                                        codes, uniques = pd.factorize(plot_df_for_shap[interaction_idx].astype(str), sort=False)
                                        plot_df_for_shap['___interaction_codes'] = codes
                                        interaction_to_pass = '___interaction_codes'
                                        friendly_labels_for_interaction = [reverse_map.get(u, u) for u in uniques]
                                except Exception:
                                    friendly_labels_for_interaction = None

                            pretty_sel = FRIENDLY_MAP.get(sel_feature, sel_feature)

                            shap.dependence_plot(plot_x_col if plot_df_for_shap is not None else sel_feature,
                                                 shap_vals_bg,
                                                 plot_df_for_shap,
                                                 feature_names=fnames_for_shap,
                                                 interaction_index=interaction_to_pass,
                                                 ax=ax,
                                                 show=False)

                            # Try to relabel the colorbar ticks with friendly names when we factorized
                            try:
                                cb_ax = fig2.axes[-1]
                                if cb_ax is not ax and sel_interaction != 'auto':
                                    try:
                                        cb_ax.set_ylabel(sel_interaction)
                                    except Exception:
                                        pass
                                    if friendly_labels_for_interaction is not None:
                                        try:
                                            n = len(friendly_labels_for_interaction)
                                            if n > 0:
                                                cb_ax.set_ticks(np.arange(n) + 0.5)
                                                cb_ax.set_yticklabels(friendly_labels_for_interaction)
                                        except Exception:
                                            try:
                                                cb_ax.set_xticklabels(friendly_labels_for_interaction)
                                            except Exception:
                                                pass
                            except Exception:
                                pass
                        except Exception:
                            # do not let plotting errors break the Explain tab
                            pass
                        # Overlay applicant's point (if available) and annotate using the helper
                        try:
                            if 'shap_vals' in locals() and shap_vals is not None and feat_names is not None:
                                if sel_feature in feat_names:
                                    idx = feat_names.index(sel_feature)
                                else:
                                    idx = None
                                if idx is not None:
                                    y_app = float(np.array(shap_vals)[0][idx])
                                    df_inp = st.session_state.get("input_df")
                                    x_app = None
                                    # Prefer to compute applicant's transformed value so it aligns with plot_df used by SHAP
                                    try:
                                        if df_inp is not None and preprocessor is not None and hasattr(preprocessor, 'transform'):
                                            try:
                                                Xt_in = preprocessor.transform(df_inp)
                                                Xt_in = Xt_in if isinstance(Xt_in, np.ndarray) else np.asarray(Xt_in)
                                                if feat_names is not None and Xt_in.shape[1] == len(feat_names):
                                                    Xt_df = pd.DataFrame(Xt_in, columns=feat_names)
                                                    if plot_x_col in Xt_df.columns:
                                                        x_app = Xt_df.iloc[0].get(plot_x_col)
                                                    elif sel_feature in Xt_df.columns:
                                                        x_app = Xt_df.iloc[0].get(sel_feature)
                                            except Exception:
                                                x_app = None

                                        # fallback to raw input values if transformed couldn't be computed
                                        if x_app is None and df_inp is not None:
                                            if plot_x_col in df_inp.columns:
                                                x_app = df_inp.iloc[0].get(plot_x_col)
                                            elif sel_feature in df_inp.columns:
                                                x_app = df_inp.iloc[0].get(sel_feature)
                                    except Exception:
                                        x_app = None

                                    if x_app is not None:
                                        _overlay_applicant_annotation(ax, x_app, y_app, df_inp, sel_feature, sel_interaction, reverse_map)
                        except Exception:
                            pass

                    plt.title(f"Dependence: {sel_feature}")
                    st.pyplot(fig2)
                    plt.close(fig2)
                except Exception as e:
                    st.warning(f"Dependence plot failed for {sel_feature}: {e}")
            except Exception as e:
                st.warning(f"Dependence plot failed for {sel_feature}: {e}")

# -------------------------
# 6. Enterprise Policy Simulator
# -------------------------
with tab3:
    st.header("🏢 Enterprise Policy Simulator")
    st.write("Act as the Chief Risk Officer. Dynamically adjust the bank's global risk tolerance and see how it impacts your active applicant.")
    
    if "input_df_raw" not in st.session_state or "proba" not in st.session_state:
        st.info("Make a prediction first under the Predict tab (top-left) to lock onto an applicant.")
    else:
        applicant_proba = st.session_state["proba"]
        
        st.subheader("Global Decision Threshold")
        threshold_sim = st.slider(
            "Set Maximum Acceptable Risk Probability", 
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Any applicant with a risk probability ABOVE this threshold will be flagged as Bad Credit."
        )
        
        if threshold_sim <= 0.30:
            st.warning("⚖️ **Policy: Highly Conservative**\n\nPrioritizes **Recall** (Catch all Defaults). You will safely reject almost all risky loans, but will aggressively sacrifice market share by falsely rejecting decent customers (False Positives).")
        elif threshold_sim <= 0.55:
            st.success("⚖️ **Policy: Balanced Growth**\n\nOptimized baseline. Balances steady market expansion while retaining acceptable default risk. Mathematically optimal for F1 Score.")
        else:
            st.error("⚖️ **Policy: Aggressive Risk (Lax)**\n\nPrioritizes **Precision** (Max Approval Volume). You only flag obvious defaults, capturing massive market share, but invite theoretically unsafe levels of hidden defaults (False Negatives).")
            
        st.markdown("---")
        st.subheader("Applicant Resolution")
        st.plotly_chart(create_threshold_bullet_chart(applicant_proba, threshold_sim), use_container_width=True)
        
        if applicant_proba >= threshold_sim:
            st.markdown(f"<div style='background-color:#fdedec; padding:15px; border-radius:8px; border-left: 5px solid red;'><h4>❌ Application Denied</h4><p>The applicant's risk probability ({applicant_proba:.3f}) exceeds the bank's maximum allowable risk threshold ({threshold_sim:.2f}).</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color:#d5f5e3; padding:15px; border-radius:8px; border-left: 5px solid green;'><h4>✅ Application Approved</h4><p>The applicant's risk probability ({applicant_proba:.3f}) safely clears the bank's maximum allowable risk threshold ({threshold_sim:.2f}).</p></div>", unsafe_allow_html=True)

# -------------------------
# 6b. Actionable Recommendations (What-If Engine)
# -------------------------
with tab4:
    st.header("💡 Actionable Recommendations")
    st.write("Simulate strategic financial changes to improve your applicant's approval odds.")
    
    if "input_df_raw" not in st.session_state:
        st.info("Make a prediction first under the Predict tab (top-left).")
    else:
        if st.button("🚀 Generate Strategic Plan"):
            with st.spinner("Analyzing heuristic risk combinations..."):
                try:
                    df_raw = st.session_state["input_df_raw"]
                    engine = RecommendationEngine(pipeline, compute_engineered_features)
                    results = engine.generate_recommendations(df_raw)
                    st.session_state["recs_results"] = results
                    st.session_state["recs_generated"] = True
                except Exception as e:
                    st.error(f"Failed to generate recommendations: {e}")

        if st.session_state.get("recs_generated", False):
            results = st.session_state["recs_results"]
            base_proba = results["base_proba"]
            base_score = results["base_score"]
            recs = results["recommendations"]
            
            st.markdown(f"### Current Estimated FICO Score: **{base_score}**")
            st.caption(f"(Based on base risk probability of {base_proba:.3f})")
            
            if not recs:
                st.success("Your applicant is highly optimized! No major heuristics triggered.")
            else:
                st.write("### Top Strategies for Optimization")
                for i, r in enumerate(recs, 1):
                    action = r['action']
                    new_score = r['new_score']
                    gain = r['points_gained']
                    new_proba = r['new_proba']
                    
                    st.markdown(f"""
                    <div style='background-color:#f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #4682b4;'>
                        <h4>{i}. {action}</h4>
                        <p style='margin:0;font-size:16px;'>Estimated New Score: <b>{new_score}</b> <span style='color:green;'><b>(+{gain} pts)</b></span></p>
                        <p style='margin:0;font-size:12px;color:#555;'>Reduces risk probability to {new_proba:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # -------------------------
                # 12-Month Score Recovery Trajectory
                # -------------------------
                st.markdown("---")
                st.subheader("📈 12-Month Score Recovery Trajectory")
                st.write("If you implement the top recommendation today and maintain perfect payment history, here is your simulated FICO score over the next 12 months.")
                
                top_new_score = recs[0]['new_score']
                # Create a logarithmic approach curve from base_score to top_new_score over 12 months
                trajectory = []
                for month in range(1, 13):
                    # Growth formula achieving ~95% of target by month 10
                    progress = 1.0 - np.exp(-0.35 * month)
                    sim_score = int(base_score + (top_new_score - base_score) * progress)
                    trajectory.append({"Month": f"Month {month:02d}", "Score": min(sim_score, 850)})
                
                traj_df = pd.DataFrame(trajectory).set_index("Month")
                st.line_chart(traj_df, use_container_width=True)

            # -------------------------
            # Live Interactive FICO Simulator
            # -------------------------
            st.markdown("---")
            st.subheader("🎛️ Live Interactive FICO Simulator")
            st.write("What happens if you rapidly change your requested financial metrics? **Drag the sliders below and watch your FICO Gauge animate in real-time.**")
            
            df_raw = st.session_state["input_df_raw"]
            current_amount = df_raw.iloc[0].get("Credit_amount", 1000)
            current_duration = df_raw.iloc[0].get("Duration_in_month", 12)
            
            sim_col1, sim_col2 = st.columns([1, 1])
            with sim_col1:
                # Add unique keys so they don't clash with Predict Tab inputs
                sim_amt = st.slider("Simulated Loan Amount ($)", min_value=250, max_value=20000, value=int(current_amount), step=50, key="sim_amt")
                sim_dur = st.slider("Simulated Loan Duration (Months)", min_value=4, max_value=72, value=int(current_duration), step=1, key="sim_dur")
                
                # Dynamically calculate the new risk probability
                df_sim = df_raw.copy()
                df_sim.at[0, "Credit_amount"] = sim_amt
                df_sim.at[0, "Duration_in_month"] = sim_dur
                
                df_sim_eng = compute_engineered_features(df_sim)
                if hasattr(pipeline_ref, "predict_proba"):
                    sim_proba = pipeline_ref.predict_proba(df_sim_eng)[0][1]
                else:
                    Xt_sim = preprocessor.transform(df_sim_eng)
                    sim_proba = classifier.predict_proba(Xt_sim)[0][1]
                    
                sim_fico = int(300 + (1.0 - sim_proba) * 550)
                
            with sim_col2:
                # Plot Real-Time Gauge
                st.plotly_chart(create_fico_gauge(sim_fico), use_container_width=True, key="sim_gauge_plot")
                
            # -------------------------
            # Generate & Download Action Plan
            # -------------------------
            st.markdown("### Generate & Download Action Plan")
            st.write("Export your customized simulated scenario directly into a physical TXT report.")
            
            amt_adj = sim_amt - current_amount
            dur_adj = sim_dur - current_duration
            amt_sign = "+" if amt_adj >= 0 else ""
            dur_sign = "+" if dur_adj >= 0 else ""
            
            rec_text = "\nRECOMMENDED OPTIMIZATION STRATEGIES:\n-----------------------------------------\n"
            if not recs:
                rec_text += "Applicant is highly optimized or well within safe limits.\n"
            else:
                for i, r in enumerate(recs, 1):
                    rec_text += f"{i}. {r['action']}\n"
                    rec_text += f"   - Estimated New Score: {r['new_score']} (+{r['points_gained']} pts)\n"

            plan_text = f"""=========================================
OPACGUARD: ADVERSE ACTION PLAN REPORT
=========================================

ORIGINAL APPLICANT REQUEST:
- Requested Loan Amount: ${current_amount}
- Requested Loan Duration: {current_duration} Months
- Baseline FICO Score: {base_score}
- Baseline Risk Probability: {base_proba:.3f}
{rec_text}
-----------------------------------------
CUSTOM SCENARIO SIMULATION (SLIDERS):
-----------------------------------------
- Adjust Loan Amount by: {amt_sign}${amt_adj} (New Total: ${sim_amt})
- Adjust Loan Duration by: {dur_sign}{dur_adj} Months (New Total: {sim_dur} Months)

- Custom Scenario FICO Score: {sim_fico} (Score Change: {sim_fico - base_score} pts)
- Custom Scenario Risk probability: {sim_proba:.3f}
"""
            st.download_button(
                label="📄 Download Action Plan (TXT)", 
                data=plan_text.encode("utf-8"), 
                file_name="OpacGuard_Action_Plan.txt", 
                mime="text/plain"
            )


    st.markdown("---")
    st.subheader("🧮 Advanced: Mathematical Counterfactuals (DiCE)")
    st.write("Calculate the absolute minimum exact numerical changes required to secure an approval (Good Credit = 0).")
    
    if st.button("Calculate Exact Requirements"):
        with st.spinner("Initializing DiCE Explainer over dataset and searching..."):
            try:
                exp = get_dice_explainer()
                df_raw = st.session_state["input_df_raw"]
                
                features_to_vary = ['Duration_in_month', 'Credit_amount']
                permitted_range = {
                    'Duration_in_month': [6, 72],
                    'Credit_amount': [250, 15000]
                }
                
                cf = exp.generate_counterfactuals(
                    df_raw, 
                    total_CFs=3, 
                    desired_class=0, # Good Credit
                    features_to_vary=features_to_vary,
                    permitted_range=permitted_range
                )
                
                cf_df = cf.cf_examples_list[0].final_cfs_df
                if cf_df is not None and len(cf_df) > 0:
                    st.success("✅ Mathematical Guaranteed Scenarios: Adjust your parameters to EXACTLY these values to secure approval.")
                    
                    original = df_raw.iloc[0]
                    for idx, row in cf_df.iterrows():
                        diffs = []
                        for col in features_to_vary:
                            if row[col] != original[col]:
                                diffs.append(f"**{col}**: {original[col]} ➡️ {row[col]}")
                        
                        if diffs:
                            st.info(" AND ".join(diffs))
                            
                else:
                    st.warning("No realistic counterfactuals could be generated within the permitted range.")
                
            except Exception as e:
                st.error(f"DiCE calculation failed: {e}")

# -------------------------
# 7. MLOps Dashboard (Admin)
# -------------------------
with tab5:
    st.header("🔐 MLOps Dashboard (Admin)")
    st.markdown("Monitor system fairness, algorithmic bias, and macroeconomic drift indicators dynamically.")
    
    st.subheader("1. Algorithmic Fairness Audit (Age Bias)")
    fairness_path = os.path.join("RESEARCH", "fairness_audit_report.txt")
    if os.path.exists(fairness_path):
        with open(fairness_path, "r") as f:
            fairness_text = f.read()
        st.code(fairness_text, language='text')
    else:
        st.info("Fairness report not found. Run RESEARCH/fairness_audit.py")
        
    st.markdown("---")
    st.subheader("2. Macroeconomic Drift Monitor (Evidently AI)")
    st.write("This interactive report displays simulated inflation macroeconomic data drift.")
    drift_path = os.path.join("RESEARCH", "drift_report.html")
    if os.path.exists(drift_path):
        with open(drift_path, "r", encoding='utf-8') as f:
            drift_html = f.read()
        import streamlit.components.v1 as components
        components.html(drift_html, height=800, scrolling=True)
    else:
        st.info("Drift HTML report not found. Run RESEARCH/drift_detection.py")

    st.markdown("---")
    st.subheader("💾 Technical Artifacts & Data Exports")
    st.write("Download raw model interpretability files and debug inputs.")
    
    colA, colB, colC = st.columns(3)
    
    pdf_path = os.path.join("results", "shap_dependence", "shap_dependence_report.pdf")
    with colA:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button("📥 SHAP & Permutation Report (PDF)", data=f, file_name="shap_dependence_report.pdf", mime="application/pdf", use_container_width=True)
        else:
            st.info("SHAP PDF not found.")
            
    with colB:
        if os.path.exists(SHAP_CSV):
            with open(SHAP_CSV, "rb") as f:
                st.download_button("📥 SHAP Global Importance (CSV)", data=f, file_name=os.path.basename(SHAP_CSV), mime="text/csv", use_container_width=True)
        else:
            st.info("SHAP CSV not found.")
            
    with colC:
        if "input_df" in st.session_state:
            csv_bytes = st.session_state["input_df"].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Last Applicant Profile (CSV)", data=csv_bytes, file_name="last_applicant_input.csv", mime="text/csv", use_container_width=True)
        else:
            st.info("Make a prediction to export applicant.")

# -------------------------
# 8. Enterprise Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888; font-size: 14px;'>OpacGuard Enterprise Risk Engine v2.4 | Core Pipeline Status: <span style='color: green;'>🟢 Nominal</span></p>", 
    unsafe_allow_html=True
)