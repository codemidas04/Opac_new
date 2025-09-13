import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

from load_data import load_german_credit

# 1. Load data
df = load_german_credit()

# Separate features & target
X = df.drop("Target", axis=1)
y = df["Target"]

# Convert target labels: 1 -> 0 (good), 2 -> 1 (bad)
y = y.map({1: 0, 2: 1})

# 2. Define categorical & numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 3. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# 4. Build model pipeline (Preprocessing + XGBoost)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        random_state=42,
        eval_metric="logloss",   # needed for XGBoost ≥1.0
        use_label_encoder=False
    ))
])

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# 6. Save trained model
joblib.dump(model, "credit_model.pkl")
print("✅ Model saved as credit_model.pkl")