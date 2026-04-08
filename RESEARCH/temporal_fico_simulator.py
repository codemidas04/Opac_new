import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from load_data import load_german_credit
from recommendation_engine.engine import score_mapping, RecommendationEngine
from model_artifact import load_model_artifact, compute_engineered_features

def generate_temporal_data():
    print("Loading base dataset...")
    df = load_german_credit()
    
    print("Loading ML pipeline...")
    pipeline = load_model_artifact()
    
    # 1. Generate Application Dates (spread over the past 3 years)
    np.random.seed(42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)
    
    days_between = (end_date - start_date).days
    random_days = np.random.randint(0, days_between, size=len(df))
    
    application_dates = [start_date + timedelta(days=int(d)) for d in random_days]
    df['application_date'] = application_dates
    
    # Sort chronologically
    df.sort_values(by='application_date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 2. Estimate FICO Scores
    print("Calculating probabilistic FICO maps for each user...")
    fico_scores = []
    probas = []
    
    # Process efficiently: we can just predict on the whole df at once
    df_features = df.drop(columns=['target', 'application_date'], errors='ignore')
    df_eng = compute_engineered_features(df_features)
    
    # We need predict_proba
    if hasattr(pipeline, "predict_proba"):
        preds = pipeline.predict_proba(df_eng)[:, 1]
    else:
        from model_artifact import find_pipeline_steps
        preprocessor, classifier, _ = find_pipeline_steps(pipeline)
        Xt = preprocessor.transform(df_eng)
        # Ensure it works even if X_trans is slightly modified
        preds = classifier.predict_proba(Xt)[:, 1]
        
    for p in preds:
        probas.append(round(float(p), 4))
        fico_scores.append(score_mapping(p))
        
    df['simulated_fico_score'] = fico_scores
    df['base_risk_probability'] = probas
    
    # Save the dataset
    out_path = os.path.join(os.path.dirname(__file__), "german_credit_with_temporal.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Generated temporal dataset with {len(df)} records.")
    print(f"✅ Saved to {out_path}")
    
    print("\nSample Output:")
    print(df[['application_date', 'Credit_amount', 'Duration_in_month', 'base_risk_probability', 'simulated_fico_score', 'target']].head(10))

if __name__ == "__main__":
    generate_temporal_data()
