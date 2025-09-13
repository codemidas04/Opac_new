import sys
import os
import joblib
import pandas as pd

# Fix import path (so Python can find load_data.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_data import load_german_credit

# Load trained model
model = joblib.load("credit_model.pkl")

# Get column names from the training dataset
df_template = load_german_credit().drop("Target", axis=1)

# Default sample (safe fallback values)
sample = {
    "Status": "A11",
    "Duration": 24,
    "CreditHistory": "A34",
    "Purpose": "A43",
    "CreditAmount": 2000,
    "Savings": "A65",
    "Employment": "A75",
    "InstallmentRate": 2,
    "PersonalStatusSex": "A93",
    "OtherDebtors": "A101",
    "ResidenceDuration": 3,
    "Property": "A121",
    "Age": 30,
    "OtherInstallmentPlans": "A143",
    "Housing": "A152",
    "ExistingCredits": 1,
    "Job": "A173",
    "LiableDependents": 1,
    "Telephone": "A192",
    "ForeignWorker": "A201"
}

print("=== Credit Risk Prediction ===")

# Let user input values interactively
for col in sample.keys():
    user_in = input(f"Enter {col} (default={sample[col]}): ").strip()
    if user_in:  # if user typed something
        # Try to cast numeric values to int/float
        if col in ["Duration", "CreditAmount", "InstallmentRate", "ResidenceDuration", "Age", "ExistingCredits", "LiableDependents"]:
            try:
                sample[col] = int(user_in)
            except ValueError:
                print(f"⚠️ Invalid number, using default {sample[col]}")
        else:
            sample[col] = user_in

# Convert to DataFrame
df = pd.DataFrame([sample])
df = df.reindex(columns=df_template.columns, fill_value=0)

# Predict
prediction = model.predict(df)[0]
proba = model.predict_proba(df)[0][1]  # probability of "bad credit"

print("\n📊 Results:")
print(f"Prediction: {'Good Credit (0)' if prediction == 0 else 'Bad Credit (1)'}")
print(f"Risk Probability (bad credit): {proba:.2f}")