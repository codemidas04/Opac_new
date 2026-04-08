import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from datetime import datetime
from model_artifact import compute_engineered_features

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_drift_detection():
    data_path = os.path.join(os.path.dirname(__file__), "german_credit_with_temporal.csv")
    if not os.path.exists(data_path):
        print("Temporal dataset not found. Run temporal_fico_simulator.py first.")
        return

    print("Loading temporal dataset...")
    df = pd.read_csv(data_path, parse_dates=['application_date'])

    # Split data chronologically (e.g. last 6 months as 'current', everything before as 'reference')
    # Because our simulator generates data ending at 'now', we can find the cutoff threshold.
    cutoff_date = df['application_date'].max() - pd.DateOffset(months=6)
    
    reference_df = df[df['application_date'] < cutoff_date].copy()
    current_df = df[df['application_date'] >= cutoff_date].copy()
    
    
    # Restrict strictly to UI approved variables and Engineered Variables
    reference_df = compute_engineered_features(reference_df)
    current_df = compute_engineered_features(current_df)

    print("\n--- Injecting Syntethic Macroeconomic Drift (Inflation) ---")
    print("Artificially inflating 'Credit_amount' in the current dataset by 25% to simulate inflation...")
    if 'Credit_amount' in current_df.columns:
        current_df['Credit_amount'] = current_df['Credit_amount'] * 1.25

    print("\nGenerating Evidently AI Data Drift Report...")
    # Generate drift report
    report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    report.run(reference_data=reference_df, current_data=current_df)
    
    out_path = os.path.join(os.path.dirname(__file__), "drift_report.html")
    report.save_html(out_path)
    
    print(f"✅ MLOps Drift report saved to {out_path}")

if __name__ == "__main__":
    run_drift_detection()
