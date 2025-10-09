"""
Export transformed feature names from the saved preprocessing pipeline to a CSV.
Usage: python backend/save_transformed_feature_names.py
Writes: results/transformed_feature_names.csv
"""
import os
import joblib
import csv

MODEL_PATHS = [
    "credit_model.pkl",
    os.path.join("..", "credit_model.pkl"),
    os.path.join("..", "backend", "credit_model.pkl"),
    os.path.join("backend", "credit_model.pkl"),
]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def find_pipeline(path):
    try:
        return joblib.load(path)
    except Exception:
        return None


def find_preprocessor(pipeline):
    if hasattr(pipeline, 'named_steps'):
        if 'preprocessor' in pipeline.named_steps:
            return pipeline.named_steps['preprocessor']
        if 'pre' in pipeline.named_steps:
            return pipeline.named_steps['pre']
        # try heuristics
        for step in pipeline.named_steps.values():
            if hasattr(step, 'transform') and not hasattr(step, 'predict'):
                return step
    else:
        if hasattr(pipeline, 'transform') and not hasattr(pipeline, 'predict'):
            return pipeline
    return None


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pre = None
    used_path = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            pl = find_pipeline(p)
            if pl is not None:
                pre = find_preprocessor(pl)
                used_path = p
                break

    if pre is None:
        print("Could not find preprocessor in any candidate pipeline. Checked:", MODEL_PATHS)
        raise SystemExit(1)

    fn = None
    try:
        if hasattr(pre, 'get_feature_names_out'):
            fn = list(pre.get_feature_names_out())
        elif hasattr(pre, 'named_steps'):
            for step in pre.named_steps.values():
                if hasattr(step, 'get_feature_names_out'):
                    fn = list(step.get_feature_names_out())
                    break
    except Exception as e:
        print('Failed to extract feature names:', e)

    if fn is None:
        print('No transformed feature names available on preprocessor')
        raise SystemExit(1)

    out_csv = os.path.join(RESULTS_DIR, 'transformed_feature_names.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['feature_transformed'])
        for name in fn:
            writer.writerow([name])

    print(f'Wrote {len(fn)} transformed feature names to {out_csv} (from {used_path})')
