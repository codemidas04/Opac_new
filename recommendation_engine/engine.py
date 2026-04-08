import pandas as pd
import numpy as np

def score_mapping(proba):
    """
    Maps probability of default (0.0 to 1.0) to a FICO-style scale (300 to 850).
    proba = 1.0 -> 300
    proba = 0.0 -> 850
    """
    score = 850 - (550 * proba)
    return int(round(score))

class RecommendationEngine:
    def __init__(self, pipeline, preprocessor_func):
        """
        pipeline: The loaded ML pipeline (must have predict_proba)
        preprocessor_func: The function to compute engineered features (e.g. compute_engineered_features)
        """
        self.pipeline = pipeline
        self.compute_engineered_features = preprocessor_func
    
    def predict_risk(self, df_raw):
        df_eng = self.compute_engineered_features(df_raw)
        # Handle cases where pipeline is just xgboost vs pipeline with preprocessor
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(df_eng)[0][1]
        else:
            # fallback if pipeline_ref structure is split for some reason, standard sklearn pipeline handles it
            from model_artifact import find_pipeline_steps
            preprocessor, classifier, _ = find_pipeline_steps(self.pipeline)
            Xt = preprocessor.transform(df_eng)
            proba = classifier.predict_proba(Xt)[0][1]
        return proba

    def generate_recommendations(self, df_raw):
        """
        Takes raw user input, calculates baseline risk, runs heuristics, and returns Top-K recommendations.
        """
        # Baseline
        base_proba = self.predict_risk(df_raw)
        base_score = score_mapping(base_proba)
        
        recommendations = []
        
        # Helper to test a variant
        def test_variant(action_name, df_variant):
            new_proba = self.predict_risk(df_variant)
            new_score = score_mapping(new_proba)
            points_gained = new_score - base_score
            if points_gained > 0:
                recommendations.append({
                    "action": action_name,
                    "new_score": new_score,
                    "points_gained": points_gained,
                    "new_proba": round(float(new_proba), 3)
                })

        # ----------------------------------------------------
        # Heuristic 1: Reduce Loan Amount
        # ----------------------------------------------------
        current_amount = df_raw.iloc[0].get("Credit_amount", 0)
        if current_amount > 1000:
            for reduction in [0.10, 0.20, 0.30]:
                df_var = df_raw.copy()
                df_var.at[0, "Credit_amount"] = current_amount * (1.0 - reduction)
                test_variant(f"Reduce requested loan amount by {int(reduction*100)}%", df_var)

        # ----------------------------------------------------
        # Heuristic 2: Shorten Duration
        # ----------------------------------------------------
        current_duration = df_raw.iloc[0].get("Duration_in_month", 0)
        if current_duration >= 12:
            for reduction_months in [6, 12]:
                new_duration = current_duration - reduction_months
                if new_duration >= 6:
                    df_var = df_raw.copy()
                    df_var.at[0, "Duration_in_month"] = new_duration
                    test_variant(f"Shorten loan duration by {reduction_months} months", df_var)

        # ----------------------------------------------------
        # Heuristic 3: Improve Checking Account (to A13: >= 200 DM)
        # ----------------------------------------------------
        current_checking = df_raw.iloc[0].get("Status_of_existing_checking_account", "")
        if current_checking in ["A11", "A12", "A14"]: 
            df_var = df_raw.copy()
            df_var.at[0, "Status_of_existing_checking_account"] = "A13"
            test_variant("Improve checking account balance to ≥200 DM", df_var)

        # ----------------------------------------------------
        # Heuristic 4: Add a Guarantor (A103)
        # ----------------------------------------------------
        current_debtors = df_raw.iloc[0].get("Other_debtors/guarantors", "")
        if current_debtors != "A103":
            df_var = df_raw.copy()
            df_var.at[0, "Other_debtors/guarantors"] = "A103"
            test_variant("Secure a qualified Guarantor", df_var)

        # ----------------------------------------------------
        # Heuristic 5: Switch from Renting to Owning Home (A152)
        # ----------------------------------------------------
        current_housing = df_raw.iloc[0].get("Housing", "")
        if current_housing == "A151": # Renting
            df_var = df_raw.copy()
            df_var.at[0, "Housing"] = "A152"
            test_variant("Acquire property asset (Home Ownership)", df_var)

        # ----------------------------------------------------
        # Heuristic 6: Increase Savings (A63 or A64)
        # ----------------------------------------------------
        current_savings = df_raw.iloc[0].get("Savings_account/bonds", "")
        if current_savings in ["A61", "A62", "A65"]:
            df_var = df_raw.copy()
            df_var.at[0, "Savings_account/bonds"] = "A64" # >= 1000 DM
            test_variant("Increase savings deposits to ≥1000 DM", df_var)

        # Sort recommendations by points gained descending
        recommendations = sorted(recommendations, key=lambda x: x["points_gained"], reverse=True)
        
        # Deduplicate similar actions if needed (e.g. reduce loan amount)
        # We will keep only the best variation of "Reduce requested loan amount"
        seen_actions = set()
        final_recs = []
        for r in recommendations:
            if "Reduce requested loan amount" in r["action"]:
                if "Reduce requested loan amount" in seen_actions:
                    continue
                seen_actions.add("Reduce requested loan amount")
            if "Shorten loan duration" in r["action"]:
                if "Shorten loan duration" in seen_actions:
                    continue
                seen_actions.add("Shorten loan duration")
            final_recs.append(r)
            if len(final_recs) == 5:
                break
                
        return {
            "base_proba": round(float(base_proba), 3),
            "base_score": base_score,
            "recommendations": final_recs
        }
