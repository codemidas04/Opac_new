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
from matplotlib.backends.backend_pdf import PdfPages
from load_data import load_german_credit, columns as ORIGINAL_COLUMNS
import json
from datetime import datetime

# -------------------------
# Utility helpers
# -------------------------
def safe_load_joblib(paths):
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError(f"Model not found. Checked: {paths}")

def find_pipeline_steps(pipeline):
    """Return (preprocessor, classifier, pipeline) if available. Handles common names."""
    pre = None
    clf = None
    if hasattr(pipeline, "named_steps"):
        # prefer explicit names if present
        if "pre" in pipeline.named_steps:
            pre = pipeline.named_steps["pre"]
        if "preprocessor" in pipeline.named_steps:
            pre = pipeline.named_steps["preprocessor"]
        if "preprocessor" not in pipeline.named_steps and pre is None:
            # try to find transformer by heuristics
            for name, step in pipeline.named_steps.items():
                if hasattr(step, "transform") and not hasattr(step, "predict"):
                    pre = step
                    break
        # classifier
        if "clf" in pipeline.named_steps:
            clf = pipeline.named_steps["clf"]
        elif "classifier" in pipeline.named_steps:
            clf = pipeline.named_steps["classifier"]
        else:
            # fallback: find first step with predict_proba or predict
            for name, step in pipeline.named_steps.items():
                if hasattr(step, "predict_proba") or hasattr(step, "predict"):
                    clf = step
                    break
    else:
        # not a pipeline - if it has predict_proba then it's a classifier
        if hasattr(pipeline, "predict_proba"):
            clf = pipeline
        if hasattr(pipeline, "transform") and not hasattr(pipeline, "predict"):
            pre = pipeline
    return pre, clf, pipeline

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


# -------------------------
# Constants
# -------------------------
MODEL_PATHS = [
    "credit_model.pkl",
    os.path.join("backend", "credit_model.pkl"),
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
# Engineered feature helper
# -------------------------
def compute_engineered_features(df_in):
    """Given a dataframe with original fields, compute the engineered features used by the pipeline
    (credit_per_month, credit_to_age, log transforms, age_bin) and return a dataframe with final order.
    This mirrors the inline code used previously but is reusable for Explain tab and tests.
    """
    df = df_in.copy()
    # ensure numeric
    df["Duration_in_month"] = pd.to_numeric(df["Duration_in_month"], errors="coerce")
    df["Credit_amount"] = pd.to_numeric(df["Credit_amount"], errors="coerce")
    df["Age_in_years"] = pd.to_numeric(df["Age_in_years"], errors="coerce")

    # engineered
    df["credit_per_month"] = df["Credit_amount"] / df["Duration_in_month"].replace(0, np.nan)
    df["credit_to_age"] = df["Credit_amount"] / df["Age_in_years"].replace(0, np.nan)
    df["log_credit_amount"] = np.log1p(df["Credit_amount"].clip(lower=0))
    df["log_duration_in_month"] = np.log1p(df["Duration_in_month"].clip(lower=0))

    def age_bin_func(age):
        if pd.isna(age):
            return np.nan
        age = float(age)
        if age < 26:
            return "18-25"
        if age < 36:
            return "26-35"
        if age < 46:
            return "36-45"
        if age < 56:
            return "46-55"
        if age < 66:
           return "56-65"
        return "65+"
    df["age_bin"] = df["Age_in_years"].apply(age_bin_func)
    # Defensive: ensure age_bin values match preprocessing labels exactly
    # (strip any accidental prefixes like 'age_bin_18-25' and coerce to the canonical set)
    try:
        df["age_bin"] = df["age_bin"].astype(str).str.replace(r"^age_bin[_-]", "", regex=True)
        # convert literal 'nan' strings back to actual NaN
        df.loc[df["age_bin"].isin(["nan", "None", "NoneType"]) , "age_bin"] = np.nan
    except Exception:
        pass

    final_cols = [
        "Status_of_existing_checking_account", "Duration_in_month", "Credit_history", "Purpose", "Credit_amount",
        "Savings_account/bonds", "Present_employment_since", "Personal_status_and_sex", "Other_debtors/guarantors",
        "Property", "Age_in_years", "Other_installment_plans", "Housing", "Job",
        "credit_per_month", "credit_to_age", "log_credit_amount", "log_duration_in_month", "age_bin"
    ]

    # ensure all final cols present (avoid KeyError) — fill missing with NaN
    for c in final_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[final_cols]


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

# Debug mode toggle (sidebar) — when enabled, show internal arrays used for SHAP waterfall
DEBUG_MODE = st.sidebar.checkbox("Debug mode", value=False)

# -------------------------
# 1. Load saved pipeline
# -------------------------
@st.cache_resource
def load_pipeline():
    possible = [
        "credit_model.pkl",
        os.path.join("backend", "credit_model.pkl")
    ]
    return safe_load_joblib(possible)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Failed to load model pipeline: {e}")
    st.stop()

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
tab1, tab2, tab3 = st.tabs(["🔮 Predict Credit Risk", "📊 Explain Prediction", "🧪 Threshold Simulator"])

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
            # Build raw-data DataFrame with exact column names the pipeline expects
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
            # keep backward compatibility if df_in is referenced elsewhere
            df_in = df_raw

            # -----------------------------
    # 2. Predict on RAW (THIS FIXES CLOUD)
    # -----------------------------
    try:
        if hasattr(pipeline_ref, "predict_proba"):
            proba = pipeline_ref.predict_proba(df_raw)[0][1]
        else:
            if preprocessor is None or classifier is None:
                raise RuntimeError("Preprocessor or classifier not found inside pipeline object.")
            Xt = preprocessor.transform(df_raw)
            proba = classifier.predict_proba(Xt)[0][1]

        label = "Bad Credit ❌" if proba >= threshold else "Good Credit ✅"

        st.session_state["proba"] = proba
        st.session_state["pred_label"] = label

        # -----------------------------
        # 3. Engineered copy ONLY for Explain tab
        # -----------------------------
        df_eng = compute_engineered_features(df_raw.copy())
        st.session_state["input_df"] = df_eng
        st.session_state["input_df_raw"] = df_raw

        st.success(f"Prediction: {label} — Risk Probability (Bad) = {proba:.3f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    # Show available prediction (if any)
    if "pred_label" in st.session_state:
        st.metric("Latest prediction", st.session_state["pred_label"])
        st.metric("Risk probability (Bad)", f"{st.session_state['proba']:.3f}")
        st.caption(f"Decision threshold used: {threshold:.2f}")

    # Debug expander with pipeline info
    with st.expander("Debug / Pipeline info", expanded=False):
        st.write("Pipeline type:", type(pipeline).__name__)
        st.write("Preprocessor:", type(preprocessor).__name__ if preprocessor is not None else None)
        st.write("Classifier:", type(classifier).__name__ if classifier is not None else None)
        try:
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                st.write("Transformed feature count:", len(preprocessor.get_feature_names_out()))
        except Exception:
            pass

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

                        # Show debug expander with the arrays used for the waterfall so you can compare (only in DEBUG_MODE)
                        if DEBUG_MODE:
                            with st.expander("Debug: SHAP waterfall inputs (feature_names, display_data, shap values)", expanded=False):
                                try:
                                    debug_payload = {
                                        "feature_names": pretty_feature_names[:min(50, len(pretty_feature_names))],
                                        "display_data_sample": display_data[:min(50, len(display_data))],
                                        "shap_vals_shape": np.array(shap_vals).shape,
                                        "shap_vals_sample": np.array(shap_vals).tolist()[:1],
                                        "base_values": basev
                                    }
                                    # Attempt to include raw transformed feature names and the transformed instance vector
                                    try:
                                        if preprocessor is not None and hasattr(preprocessor, 'get_feature_names_out'):
                                            raw_names = list(preprocessor.get_feature_names_out())
                                            debug_payload['transformed_feature_names_raw'] = raw_names[:min(200, len(raw_names))]
                                        # transformed instance vector
                                        try:
                                            Xt_in = preprocessor.transform(df_row)
                                            Xt_arr = Xt_in if isinstance(Xt_in, (np.ndarray,)) else np.asarray(Xt_in)
                                            # convert to python lists for nicer display
                                            debug_payload['transformed_instance_vector'] = Xt_arr.tolist()[0] if Xt_arr.ndim == 2 else Xt_arr.tolist()
                                        except Exception as e:
                                            debug_payload['transformed_instance_vector_error'] = str(e)
                                    except Exception:
                                        pass
                                    st.write(debug_payload)
                                    # Persist debug payload to results for offline inspection
                                    try:
                                        os.makedirs(RESULTS_DIR, exist_ok=True)
                                        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                                        out_path = os.path.join(RESULTS_DIR, f"shap_debug_{ts}.json")
                                        with open(out_path, "w", encoding="utf-8") as _jf:
                                            json.dump(debug_payload, _jf, default=str, indent=2)
                                        st.caption(f"Saved debug payload to {out_path}")
                                    except Exception:
                                        pass
                                except Exception as e:
                                    st.write("Could not render debug arrays:", e)

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
# 6. Threshold Simulator
# -------------------------
with tab3:
    st.header("🧪 Threshold Simulator")
    st.write("Simulate how a fixed risk score would be classified under different thresholds.")
    prob_sim = st.slider("Simulated risk probability (Bad)", 0.0, 1.0, 0.45, 0.01)
    thresholds = st.multiselect("Choose thresholds", [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], default=[0.35, 0.5])
    if thresholds:
        rows = [{"Threshold": t, "Prediction": ("Bad Credit ❌" if prob_sim >= t else "Good Credit ✅")} for t in thresholds]
        st.table(pd.DataFrame(rows))

# -------------------------
# 7. Reports
# -------------------------
st.header("📎 Reports & Artifacts")
pdf_path = os.path.join("results", "shap_dependence", "shap_dependence_report.pdf")
if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download SHAP + Permutation Report (PDF)", data=f, file_name="shap_dependence_report.pdf", mime="application/pdf")
else:
    st.info("No SHAP report PDF found at results/shap_dependence/shap_dependence_report.pdf — run backend/shap_avg.py to generate it.")

# Additional downloads: SHAP CSV and last input
if os.path.exists(SHAP_CSV):
    with open(SHAP_CSV, "rb") as f:
        st.download_button("📥 Download SHAP mean-abs importance (CSV)", data=f, file_name=os.path.basename(SHAP_CSV), mime="text/csv")

if "input_df" in st.session_state:
    csv_bytes = st.session_state["input_df"].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download last input (CSV)", data=csv_bytes, file_name="last_input.csv", mime="text/csv")

# -------------------------
# 8. Footer
# -------------------------
st.markdown("---")
st.markdown(
    "ℹ️ Notes:\n\n"
    "- The UI collects original dataset fields; engineered features are computed automatically.\n"
    "- If SHAP plots fail, ensure the saved pipeline exposes a preprocessor (ColumnTransformer/Pipeline) and a classifier.\n"
    "- If you want the top-N SHAP features precomputed and prettier labels, we can extend the friendly mapping file."
)