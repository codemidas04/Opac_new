import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
from tabulate import tabulate   # ✅ for pretty console tables

from load_data import load_german_credit

# -----------------------------
# Save directory
# -----------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Load & preprocess
# -----------------------------
df = load_german_credit()
X = df.drop("Target", axis=1)
y = df["Target"].map({1: 0, 2: 1})

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Helper: train + evaluate
# -----------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test, results):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results.append({
        "Model": name,
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1-score": round(f1, 2),
        "AUC-ROC": round(auc, 2) if auc is not None else "N/A"
    })

    return y_proba

# -----------------------------
# Models
# -----------------------------
results = []
probas = {}

# Dummy
dummy = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", DummyClassifier(strategy="most_frequent"))
])
evaluate_model("Dummy (Most Frequent)", dummy, X_train, y_train, X_test, y_test, results)

# Logistic Regression
log_reg = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=2000))
])
probas["Logistic Regression"] = evaluate_model("Logistic Regression", log_reg, X_train, y_train, X_test, y_test, results)

# Decision Tree
tree = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", DecisionTreeClassifier(random_state=42))
])
probas["Decision Tree"] = evaluate_model("Decision Tree", tree, X_train, y_train, X_test, y_test, results)

# XGBoost (default)
xgb = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", XGBClassifier(random_state=42, eval_metric="logloss"))
])
probas["XGBoost (Default)"] = evaluate_model("XGBoost (Default)", xgb, X_train, y_train, X_test, y_test, results)

# -----------------------------
# XGBoost (tuned with RandomizedSearchCV)
# -----------------------------
xgb_tuned = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", XGBClassifier(random_state=42, eval_metric="logloss"))
])

param_dist = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [3, 4, 5, 6],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "clf__subsample": [0.7, 0.8, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 1.0],
    "clf__gamma": [0, 1, 5]
}

random_search = RandomizedSearchCV(
    xgb_tuned,
    param_distributions=param_dist,
    n_iter=10,
    scoring="f1",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_xgb = random_search.best_estimator_
print(f"\n✅ Best XGBoost Params: {random_search.best_params_}")

probas["XGBoost (Tuned)"] = evaluate_model("XGBoost (Tuned)", best_xgb, X_train, y_train, X_test, y_test, results)

# -----------------------------
# Plot ROC & PR curves
# -----------------------------
plt.figure(figsize=(12, 5))

# ROC
plt.subplot(1, 2, 1)
for name, y_proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend()

# Precision-Recall
plt.subplot(1, 2, 2)
for name, y_proba in probas.items():
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(rec, prec, label=name)
plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "model_comparison_curves.png")
plt.savefig(plot_path, dpi=300)
plt.show(block=False)   # ✅ non-blocking
print(f"\n📊 Saved ROC & PR curves to {plot_path}")

# -----------------------------
# Save summary + pretty print
# -----------------------------
summary_df = pd.DataFrame(results)
summary_path = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"📊 Saved summary table to {summary_path}")

# Highlight best values
best_vals = {}
for col in ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]:
    valid_vals = [v for v in summary_df[col] if v != "N/A"]
    if valid_vals:
        best_vals[col] = max(valid_vals)

def colorize(val, col):
    if val == "N/A":
        return val
    return f"\033[92m\033[1m{val:.2f}\033[0m" if val == best_vals[col] else f"{val:.2f}"

table = []
for _, row in summary_df.iterrows():
    table.append([
        row["Model"],
        colorize(row["Accuracy"], "Accuracy"),
        colorize(row["Precision"], "Precision"),
        colorize(row["Recall"], "Recall"),
        colorize(row["F1-score"], "F1-score"),
        colorize(row["AUC-ROC"], "AUC-ROC"),
    ])

print("\n📊 Model Comparison Summary:")
print(tabulate(table, headers=["Model", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"], tablefmt="pretty"))