import os
import joblib

def load_model_artifact(possible_paths=None):
    """
    Intelligently resolves paths so the model loads flawlessly whether the app is run from 
    the project root or the backend/ directory.
    """
    if possible_paths is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(base_dir, "credit_model.pkl"),
            os.path.join(base_dir, "backend", "credit_model.pkl"),
            os.path.join(base_dir, "..", "credit_model.pkl"),
            "credit_model.pkl",
            "backend/credit_model.pkl"
        ]
        
    for p in possible_paths:
        if os.path.exists(p):
            return joblib.load(p)
            
    raise FileNotFoundError(f"credit_model.pkl not found. Checked: {possible_paths}")

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

def get_column_transformer(preprocessor):
    """
    Introspects the preprocessor to find the ColumnTransformer.
    Handles cases where preprocessor is a Pipeline (e.g. FeatureEngineering -> ColumnTransformer).
    """
    import sklearn
    from sklearn.compose import ColumnTransformer
    
    if isinstance(preprocessor, ColumnTransformer):
        return preprocessor
        
    if hasattr(preprocessor, "named_steps"):
        for name, step in preprocessor.named_steps.items():
            if isinstance(step, ColumnTransformer):
                return step
                
    if hasattr(preprocessor, "transformers_"):
        return preprocessor
        
    return None

import pandas as pd
import numpy as np

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
