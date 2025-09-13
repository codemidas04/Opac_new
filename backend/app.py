import os
import traceback
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import pandas as pd
import shap

app = Flask(__name__)
# Allow requests from React dev server
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load model robustly (path next to app.py)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "credit_model.pkl")
model = joblib.load(MODEL_PATH)

REQUIRED_FEATURES = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "Employment", "InstallmentRate", "PersonalStatusSex",
    "OtherDebtors", "ResidenceDuration", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "ExistingCredits",
    "Job", "LiableDependents", "Telephone", "ForeignWorker"
]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Credit Risk API is running"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        missing = [f for f in REQUIRED_FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "label": "Bad Credit" if prediction == 1 else "Good Credit",
            "risk_probability": round(float(proba), 2)
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500

@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json()
        missing = [f for f in REQUIRED_FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        df = pd.DataFrame([data])

        # --- Step 1: Preprocess input (OneHot + numeric passthrough)
        preprocessed = model.named_steps["preprocessor"].transform(df)

        # --- Step 2: Get feature names from preprocessor
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        # --- Step 2b: Build mapping for categorical features
        encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
        cat_cols = encoder.feature_names_in_
        categories = encoder.categories_

        mapping = {}
        for col, cats in zip(cat_cols, categories):
            for cat in cats:
                mapping[f"cat__{col}_{cat}"] = f"{col} = {cat}"

        # Numeric passthrough features (strip num__ prefix)
        for col in model.named_steps["preprocessor"].transformers_[1][2]:
            mapping[f"num__{col}"] = col

        # --- Step 3: Run SHAP on the classifier
        explainer = shap.Explainer(model.named_steps["classifier"])
        shap_values = explainer(preprocessed)

        # --- Step 4: Pick top 5 contributing features
        raw_contributions = dict(
            sorted(
                zip(feature_names, shap_values.values[0]),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
        )

        # Map feature names to clean labels
        contributions = {mapping.get(k, k): float(v) for k, v in raw_contributions.items()}

        # --- Step 5: Get prediction + probability
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "label": "Bad Credit" if prediction == 1 else "Good Credit",
            "risk_probability": round(float(proba), 2),
            "top_features": contributions
        })

    except Exception as e:
        return jsonify({
            "error": "Explainability failed",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500
if __name__ == "__main__":
    app.run(debug=True)