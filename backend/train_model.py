import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # keep root on path

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from xgboost import XGBClassifier
import joblib
from datetime import datetime

from load_data import load_german_credit

# -----------------------------
# CLI args
# -----------------------------
parser = argparse.ArgumentParser(description="Train Credit Risk Model")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for classifying Bad Credit (default 0.5)")
parser.add_argument("--skip-tuning", action="store_true",
                    help="Skip hyperparameter tuning and use last known best params")
args = parser.parse_args()
chosen_threshold = args.threshold

# -----------------------------
# Results directory
# -----------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# Load & preprocess data
# -----------------------------
df = load_german_credit()
X = df.drop("Target", axis=1)
y = df["Target"].map({1: 0, 2: 1})  # Good=0, Bad=1

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# -----------------------------
# Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Build pipeline
# -----------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

# -----------------------------
# Hyperparameter tuning
# -----------------------------
if not args.skip_tuning:
    print("\n🔍 Running hyperparameter tuning (RandomizedSearchCV)...")

    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__gamma": [0, 1, 5],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"\n✅ Best XGBoost Params: {search.best_params_}")

else:
    print("\n⏩ Skipping tuning, using default XGBoost...")
    model = pipeline
    model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete.")

y_proba = model.predict_proba(X_test)[:, 1]    # probability of class "Bad Credit"
y_pred_default = (y_proba >= 0.5).astype(int)

# -----------------------------
# Metrics at default threshold
# -----------------------------
acc = accuracy_score(y_test, y_pred_default)
prec = precision_score(y_test, y_pred_default, zero_division=0)
rec = recall_score(y_test, y_pred_default, zero_division=0)
f1 = f1_score(y_test, y_pred_default, zero_division=0)
auc_roc = roc_auc_score(y_test, y_proba)

print("\n📊 Model Performance Metrics (Default Threshold 0.5)")
print(f"Accuracy:   {acc:.2f}")
print(f"Precision:  {prec:.2f}")
print(f"Recall:     {rec:.2f}")
print(f"F1-score:   {f1:.2f}")
print(f"AUC-ROC:    {auc_roc:.2f}")
print("Confusion Matrix (Default):")
print(confusion_matrix(y_test, y_pred_default))
print("\nClassification Report (Default):")
print(classification_report(y_test, y_pred_default, target_names=["Good Credit", "Bad Credit"], zero_division=0))

# -----------------------------
# Metrics at tuned threshold
# -----------------------------
y_pred_tuned = (y_proba >= chosen_threshold).astype(int)
acc_t = accuracy_score(y_test, y_pred_tuned)
prec_t = precision_score(y_test, y_pred_tuned, zero_division=0)
rec_t = recall_score(y_test, y_pred_tuned, zero_division=0)
f1_t = f1_score(y_test, y_pred_tuned, zero_division=0)

print(f"\n📊 Model Performance Metrics (Tuned Threshold {chosen_threshold})")
print(f"Accuracy:   {acc_t:.2f}")
print(f"Precision:  {prec_t:.2f}")
print(f"Recall:     {rec_t:.2f}")
print(f"F1-score:   {f1_t:.2f}")
print("Confusion Matrix (Tuned):")
print(confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report (Tuned):")
print(classification_report(y_test, y_pred_tuned, target_names=["Good Credit", "Bad Credit"], zero_division=0))

# -----------------------------
# ROC + Precision-Recall plots
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)

precisions, recalls, _ = precision_recall_curve(y_test, y_proba)

# Compute point for chosen_threshold
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
tn, fp, fn, tp = cm_tuned.ravel()
fpr_tuned = fp / (fp + tn) if (fp + tn) > 0 else 0.0
tpr_tuned = tp / (tp + fn) if (tp + fn) > 0 else 0.0
prec_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
rec_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)

plt.figure(figsize=(12, 5))

# ROC
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc_val:.2f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.scatter([fpr_tuned], [tpr_tuned], color="red", s=100, zorder=5, label=f"Threshold = {chosen_threshold}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")

# Precision-Recall
plt.subplot(1, 2, 2)
plt.plot(recalls, precisions, color="blue", lw=2)
plt.scatter([rec_tuned], [prec_tuned], color="red", s=100, zorder=5, label=f"Threshold = {chosen_threshold}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")

plt.tight_layout()

# Save plots to results/
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join(RESULTS_DIR, f"train_curves_{ts}.png")
plt.savefig(plot_path, dpi=300)
print(f"\n📊 Saved ROC & PR curves to {plot_path}")
plt.show(block=False)

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "credit_model.pkl")
print("\n✅ Model saved as credit_model.pkl")