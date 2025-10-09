# oof_train_shap.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
from preprocessing import build_preprocessor
from load_data import load_german_credit
import os, json

# load
df = load_german_credit()
X = df.drop('target', axis=1)
y = df['target'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
models = []

# best params found earlier (map to XGBClassifier args)
best_params = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=1.0,
    gamma=1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    pre = build_preprocessor()
    clf = Pipeline([('pre', pre), ('clf', XGBClassifier(**best_params))])

    # early stopping: pass full eval_set via fit kwargs
    clf.fit(X_tr, y_tr)
    oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    models.append(clf)
    print(f"Fold {fold} done. Val AUC: {roc_auc_score(y_val, oof[val_idx]):.4f}")

# overall OOF AUC
oof_auc = roc_auc_score(y, oof)
print("OOF AUC:", oof_auc)

# save OOF preds and a simple metadata file
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame({'oof_proba': oof, 'target': y}).to_csv(os.path.join(out_dir, 'oof_preds.csv'), index=False)
with open(os.path.join(out_dir, 'oof_meta.json'), 'w') as f:
    json.dump({'oof_auc': float(oof_auc), 'best_params': best_params}, f, indent=2)

# Optionally save an ensemble of fold models
joblib.dump(models, os.path.join(out_dir, 'fold_models.pkl'))
print("Saved OOF preds and fold models.")