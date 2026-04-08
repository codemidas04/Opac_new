import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import dice_ml
from dice_ml.utils import helpers
from load_data import load_german_credit
from model_artifact import load_model_artifact, compute_engineered_features

def run_counterfactual_generator():
    print("Loading Data and Model...")
    df = load_german_credit()
    pipeline = load_model_artifact()

    # DiCE needs to know which features can be mutated. We define raw features here.
    continuous_features = ['Duration_in_month', 'Credit_amount', 'Age_in_years']
    
    # Target definition
    target = 'target'
    
    # To use DiCE, we must provide a predict function that takes a raw dataframe and outputs predictions.
    # We will create a custom wrapper class for sklearn compatibility.
    class CustomWrapper:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def predict_proba(self, X):
            X_copy = X.copy()
            # Engineering step
            X_eng = compute_engineered_features(X_copy)
            
            if hasattr(self.pipeline, "predict_proba"):
                return self.pipeline.predict_proba(X_eng)
            else:
                from model_artifact import find_pipeline_steps
                preprocessor, classifier, _ = find_pipeline_steps(self.pipeline)
                Xt = preprocessor.transform(X_eng)
                return classifier.predict_proba(Xt)
                
        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

    wrapped_model = CustomWrapper(pipeline)

    print("Configuring DiCE...")
    # Initialize DiCE Data
    d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name=target)
    
    # Initialize DiCE Model (backend=sklearn for general python predict interfaces)
    m = dice_ml.Model(model=wrapped_model, backend='sklearn')
    
    # Initialize DiCE Explainer
    exp = dice_ml.Dice(d, m, method="random")

    # Pick a high-risk applicant to test
    # A user predicted as Bad Credit (1)
    df_raw = df.drop(columns=[target])
    preds = wrapped_model.predict(df_raw)
    
    high_risk_indices = np.where(preds == 1)[0]
    if len(high_risk_indices) > 0:
        query_idx = high_risk_indices[0]
        query_instances = df_raw.iloc[[query_idx]]
        
        print(f"\n--- Testing Query Instance (Index {query_idx}) ---")
        print("Original Prediction: BAD CREDIT (Risk Class 1)")
        print("\nSearching for 3 minimal counterfactuals to flip prediction to GOOD CREDIT (Risk Class 0)...")
        
        # permitted range dict to ensure realistic constraints
        features_to_vary = ['Duration_in_month', 'Credit_amount']
        permitted_range = {
            'Duration_in_month': [6, 72],
            'Credit_amount': [250, 15000]
        }
        
        dice_exp = exp.generate_counterfactuals(
            query_instances, 
            total_CFs=3, 
            desired_class=0, # Good Credit
            features_to_vary=features_to_vary,
            permitted_range=permitted_range
        )
        
        dice_exp.visualize_as_dataframe(show_only_changes=True)
        
        out_path = os.path.join(os.path.dirname(__file__), "counterfactual_results")
        import json
        cf_json = dice_exp.to_json()
        with open(out_path + ".json", 'w') as f:
            f.write(cf_json)
            
        print(f"\n✅ Counterfactuals mathematically isolated and saved to {out_path}.json")
    else:
        print("No high risk applicants found in the dataset to test counterfactuals.")

if __name__ == "__main__":
    run_counterfactual_generator()
