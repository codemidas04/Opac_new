# preprocessing.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# --- Custom transformer for feature engineering ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ratio features
        if "Credit_amount" in X and "Duration_in_month" in X:
            X["credit_per_month"] = X["Credit_amount"] / X["Duration_in_month"].replace(0, np.nan)
        if "Credit_amount" in X and "Age_in_years" in X:
            X["credit_to_age"] = X["Credit_amount"] / X["Age_in_years"].replace(0, np.nan)

        # Log transforms
        if "Credit_amount" in X:
            X["log_credit_amount"] = np.log1p(X["Credit_amount"])
        if "Duration_in_month" in X:
            X["log_duration_in_month"] = np.log1p(X["Duration_in_month"])

        # Age binning
        if "Age_in_years" in X:
            X["age_bin"] = pd.cut(
                X["Age_in_years"],
                bins=[18, 25, 35, 45, 55, 65, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
                right=False
            )

        # Drop weak/redundant columns
        drop_cols = [
            "Telephone",
            "Foreign_worker",
            "Number_of_people_being_liable_to_provide_maintenance_for",
            "Installment_rate_in_percentage_of_disposable_income",
            "Present_residence_since",
            "Number_of_existing_credits_at_this_bank",
            "Credit_risk"  # keep only 'target'
        ]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

        return X


# --- Build preprocessing pipeline ---
def build_preprocessor():
    categorical_cols = [
        "Status_of_existing_checking_account", "Credit_history", "Purpose",
        "Savings_account/bonds", "Present_employment_since",
        "Personal_status_and_sex", "Other_debtors/guarantors", "Property",
        "Other_installment_plans", "Housing", "Job", "age_bin"
    ]

    numeric_cols = [
        "Duration_in_month", "Credit_amount", "Age_in_years",
        "credit_per_month", "credit_to_age",
        "log_credit_amount", "log_duration_in_month"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Full pipeline: first engineer features, then preprocess
    full_pipeline = Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline
