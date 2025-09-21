import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # keep root on path

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

from load_data import load_german_credit

# -----------------------------
# CLI args
# -----------------------------
parser = argparse.ArgumentParser(description="Train Credit Risk Model")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for classifying Bad Credit (default 0.5)")
args = parser.parse_args()
chosen_threshold = args.threshold

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

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ))
])

# -----------------------------
# Train / Test split & fit
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
model.fit(X_train, y_train)
print("Training complete.")

# -----------------------------
# Predicted probabilities & default predictions (0.5)
# -----------------------------
y_proba = model.predict_proba(X_test)[:, 1]    # probability of class "Bad Credit"
y_pred_default = (y_proba >= 0.5).astype(int)

# -----------------------------
# Metrics at default threshold = 0.5
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
# Metrics at tuned threshold (CLI arg)
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
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
roc_auc_val = auc(fpr, tpr)

precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)

# compute point for chosen_threshold (use confusion matrix-based point)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
tn, fp, fn, tp = cm_tuned.ravel()
fpr_tuned = fp / (fp + tn) if (fp + tn) > 0 else 0.0
tpr_tuned = tp / (tp + fn) if (tp + fn) > 0 else 0.0
prec_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
rec_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)

# Plot side-by-side
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
plt.show()

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "credit_model.pkl")
print("\n✅ Model saved as credit_model.pkl")