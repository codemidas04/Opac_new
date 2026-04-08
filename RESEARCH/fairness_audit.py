import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, MetricFrame
from sklearn.metrics import accuracy_score, precision_score
from load_data import load_german_credit
from model_artifact import load_model_artifact, compute_engineered_features

def run_fairness_audit():
    print("Loading Data and Model for Fairness Audit...")
    df = load_german_credit()
    pipeline = load_model_artifact()

    # Pre-process raw data exactly as Streamlit does
    X_raw = df.drop(columns=['target'])
    y_true = df['target']
    
    X_eng = compute_engineered_features(X_raw)
    
    if hasattr(pipeline, "predict_proba"):
        preds_proba = pipeline.predict_proba(X_eng)[:, 1]
    else:
        from model_artifact import find_pipeline_steps
        preprocessor, classifier, _ = find_pipeline_steps(pipeline)
        Xt = preprocessor.transform(X_eng)
        preds_proba = classifier.predict_proba(Xt)[:, 1]
        
    # We will use the standard default threshold of 0.5 to declare 'Bad Credit' (1)
    y_pred = (preds_proba >= 0.5).astype(int)

    print("\n--- Fairness Metrics on Protected Attribute: AGE (<25 vs >=25) ---")
    
    # German Credit has 'Age_in_years'. We binarize it for the audit: < 25 is "Young" (potentially discriminated against).
    sensitive_feature = (df['Age_in_years'] < 25).map({True: 'Young (< 25)', False: 'Older (>= 25)'})

    # The favorable outcome is 'Good Credit', which in our target mapping is 0. 
    # But fairlearn often treats 1 as favorable. 
    # Let's invert temporarily for fairlearn so 1 = Good Credit (Approval).
    y_true_fav = 1 - y_true
    y_pred_fav = 1 - y_pred

    mf = MetricFrame(
        metrics={
            'approval_rate': lambda y_t, y_p: np.mean(y_p),
            'accuracy': accuracy_score
        },
        y_true=y_true_fav,
        y_pred=y_pred_fav,
        sensitive_features=sensitive_feature
    )
    
    print("\nGroup Metrics:")
    print(mf.by_group)
    
    # Disparate impact ratio is the ratio of approval rates
    dp_ratio = demographic_parity_ratio(y_true_fav, y_pred_fav, sensitive_features=sensitive_feature)
    dp_diff = demographic_parity_difference(y_true_fav, y_pred_fav, sensitive_features=sensitive_feature)
    
    print(f"\nDemographic Parity Difference: {dp_diff:.4f}")
    print(f"Demographic Parity Ratio: {dp_ratio:.4f}")
    
    if dp_ratio < 0.8:
        print("\n⚠️ WARNING: Demographic Parity Ratio is below the 0.8 (Four-Fifths) standard.")
        print("This indicates potential algorithmic bias (Disparate Impact) against the young demographic.")
    else:
        print("\n✅ passes the 0.8 Four-Fifths rule for Disparate Impact.")

    out_path = os.path.join(os.path.dirname(__file__), "fairness_audit_report.txt")
    with open(out_path, "w") as f:
        f.write("FAIRNESS AUDIT REPORT (fairlearn)\n")
        f.write("="*50 + "\n")
        f.write(mf.by_group.to_string() + "\n\n")
        f.write(f"Demographic Parity Difference: {dp_diff:.4f}\n")
        f.write(f"Demographic Parity Ratio (Disparate Impact): {dp_ratio:.4f}\n")

    print(f"\n✅ Fairness audit report saved to {out_path}")

if __name__ == "__main__":
    run_fairness_audit()
