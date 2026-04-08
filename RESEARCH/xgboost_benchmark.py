import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from load_data import load_german_credit
from model_artifact import compute_engineered_features, find_pipeline_steps
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from preprocessing import build_preprocessor

def run_benchmark():
    print("Loading base dataset for Benchmark...")
    df = load_german_credit()
    X = df.drop('target', axis=1)
    y = df['target'].values
    
    # Stratified KFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_lr = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 1.0,
        "gamma": 1,
        "random_state": 42,
        "eval_metric": 'logloss'
    }
    
    lr_params = {
        "max_iter": 1000,
        "random_state": 42
    }
    
    print("\nStarting 5-Fold Stratified Cross-Validation Benchmark...")
    print("-" * 50)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        # We need to engineer features on the splits physically before sending them to sklearn pipelines
        # Usually pipelines do this, but since we separated feature engineering to pure pandas for Streamlit predictability:
        # We process X_tr and X_val via compute_engineered_features. 
        # But wait, our `build_preprocessor()` expects engineered columns explicitly.
        
        X_tr_eng = compute_engineered_features(X_tr)
        X_val_eng = compute_engineered_features(X_val)
        
        pre = build_preprocessor()
        
        # Logistic Regression
        Xt_tr = pre.fit_transform(X_tr_eng)
        Xt_val = pre.transform(X_val_eng)
        
        lr = LogisticRegression(**lr_params)
        lr.fit(Xt_tr, y_tr)
        oof_lr[val_idx] = lr.predict_proba(Xt_val)[:, 1]
        
        # XGBoost
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(Xt_tr, y_tr)
        oof_xgb[val_idx] = xgb.predict_proba(Xt_val)[:, 1]
        
        lr_fold_auc = roc_auc_score(y_val, oof_lr[val_idx])
        xgb_fold_auc = roc_auc_score(y_val, oof_xgb[val_idx])
        
        print(f"Fold {fold} | LogisticRegression AUC: {lr_fold_auc:.4f} | XGBoost AUC: {xgb_fold_auc:.4f}")

    print("-" * 50)
    auc_lr_total = roc_auc_score(y, oof_lr)
    auc_xgb_total = roc_auc_score(y, oof_xgb)
    
    print(f"\n[FINAL RESULTS] LogisticRegression OOF AUC: {auc_lr_total:.4f}")
    print(f"[FINAL RESULTS] XGBoost OOF AUC:            {auc_xgb_total:.4f}")
    
    # Save the report dataset
    benchmark_df = pd.DataFrame({
        "LogisticRegression_Proba": oof_lr,
        "XGBoost_Proba": oof_xgb,
        "True_Target": y
    })
    
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
    benchmark_df.to_csv(out_path, index=False)
    print(f"\n✅ Benchmark results saved to {out_path}")

if __name__ == "__main__":
    run_benchmark()
