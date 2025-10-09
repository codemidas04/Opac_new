# shap_avg.py (enhanced with readable labels + SHAP plots for top-N features)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from load_data import load_german_credit
from sklearn.inspection import permutation_importance

# ============================================
# Feature Code → Human-Readable Mapping
# ============================================
code_map = {
    # --- Engineered features ---
    "credit_per_month": "Credit per Month",
    "credit_to_age": "Credit-to-Age Ratio",
    "log_credit_amount": "Log(Credit Amount)",
    "log_duration_in_month": "Log(Duration in Months)",
    "age_bin_18-25": "Age Bin: 18–25",
    "age_bin_26-35": "Age Bin: 26–35",
    "age_bin_36-45": "Age Bin: 36–45",
    "age_bin_46-55": "Age Bin: 46–55",
    "age_bin_56-65": "Age Bin: 56–65",
    "age_bin_65+": "Age Bin: 65+",

    # --- Original numeric features ---
    "Duration_in_month": "Duration (Months)",
    "Credit_amount": "Credit Amount",
    "Age_in_years": "Age (Years)",

    # --- Status of existing checking account (A11–A14) ---
    "Status_of_existing_checking_account_A11": "Checking: <0 DM",
    "Status_of_existing_checking_account_A12": "Checking: 0–200 DM",
    "Status_of_existing_checking_account_A13": "Checking: ≥200 DM",
    "Status_of_existing_checking_account_A14": "Checking: None",

    # --- Credit history (A30–A34) ---
    "Credit_history_A30": "History: No Credit Taken",
    "Credit_history_A31": "History: All Paid Duly",
    "Credit_history_A32": "History: Existing Paid (Delay)",
    "Credit_history_A33": "History: Existing Paid (Other)",
    "Credit_history_A34": "History: Critical/Other Loans",

    # --- Purpose (A40–A49) ---
    "Purpose_A40": "Purpose: New Car",
    "Purpose_A41": "Purpose: Used Car",
    "Purpose_A42": "Purpose: Furniture",
    "Purpose_A43": "Purpose: Radio/TV",
    "Purpose_A44": "Purpose: Appliances",
    "Purpose_A45": "Purpose: Repairs",
    "Purpose_A46": "Purpose: Education",
    "Purpose_A47": "Purpose: Vacation",
    "Purpose_A48": "Purpose: Retraining",
    "Purpose_A49": "Purpose: Business",
    "Purpose_A410": "Purpose: Other",

    # --- Savings account/bonds (A61–A65) ---
    "Savings_account/bonds_A61": "Savings: <100 DM",
    "Savings_account/bonds_A62": "Savings: 100–500 DM",
    "Savings_account/bonds_A63": "Savings: 500–1000 DM",
    "Savings_account/bonds_A64": "Savings: ≥1000 DM",
    "Savings_account/bonds_A65": "Savings: Unknown",

    # --- Present employment since (A71–A75) ---
    "Present_employment_since_A71": "Employment: Unemployed",
    "Present_employment_since_A72": "Employment: <1 yr",
    "Present_employment_since_A73": "Employment: 1–4 yrs",
    "Present_employment_since_A74": "Employment: 4–7 yrs",
    "Present_employment_since_A75": "Employment: ≥7 yrs",

    # --- Personal status & sex (A91–A95) ---
    "Personal_status_and_sex_A91": "Male: Divorced/Separated",
    "Personal_status_and_sex_A92": "Female: Single",
    "Personal_status_and_sex_A93": "Male: Single",
    "Personal_status_and_sex_A94": "Male: Married/Widowed",
    "Personal_status_and_sex_A95": "Female: Divorced/Married",

    # --- Other debtors / guarantors (A101–A103) ---
    "Other_debtors/guarantors_A101": "Debtors: None",
    "Other_debtors/guarantors_A102": "Debtors: Co-Applicant",
    "Other_debtors/guarantors_A103": "Debtors: Guarantor",

    # --- Property (A121–A124) ---
    "Property_A121": "Property: Real Estate",
    "Property_A122": "Property: Insurance",
    "Property_A123": "Property: Car (Other)",
    "Property_A124": "Property: None",

    # --- Other installment plans (A141–A143) ---
    "Other_installment_plans_A141": "Installment: Bank",
    "Other_installment_plans_A142": "Installment: Stores",
    "Other_installment_plans_A143": "Installment: None",

    # --- Housing (A151–A153) ---
    "Housing_A151": "Housing: Rent",
    "Housing_A152": "Housing: Own",
    "Housing_A153": "Housing: Free",

    # --- Job (A171–A174) ---
    "Job_A171": "Job: Unskilled Non-Resident",
    "Job_A172": "Job: Unskilled Resident",
    "Job_A173": "Job: Skilled",
    "Job_A174": "Job: Highly Skilled",

    # --- Telephone (A191–A192) ---
    "Telephone_A191": "Telephone: None",
    "Telephone_A192": "Telephone: Yes",

    # --- Foreign worker (A201–A202) ---
    "Foreign_worker_A201": "Foreign Worker: Yes",
    "Foreign_worker_A202": "Foreign Worker: No",
}

# ========================================
# 2. Load fold models & dataset
# ========================================
models = joblib.load("results/fold_models.pkl")
df = load_german_credit()
X = df.drop("target", axis=1)
y = df["target"].values

imp_list = []
all_feat_names = None

# ========================================
# 3. Collect SHAP values across folds
# ========================================
for i, clf in enumerate(models, 1):
    print(f"Computing SHAP for fold {i}...")
    pre = clf.named_steps["pre"]
    model = clf.named_steps["clf"]

    X_trans = pre.transform(X)

    try:
        col_transformer = pre.named_steps["preprocessor"]
        feat_names = col_transformer.get_feature_names_out()
        feat_names = [name.split("__")[-1] for name in feat_names]
    except Exception as e:
        print("⚠️ Could not extract feature names:", e)
        feat_names = [f"f{j}" for j in range(X_trans.shape[1])]

    if all_feat_names is None:
        all_feat_names = feat_names

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_trans)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    imp_list.append(mean_abs)

# Average SHAP across folds
imp_avg = np.mean(np.vstack(imp_list), axis=0)

# Map to friendly labels
imp_df = pd.DataFrame({
    "feature_raw": all_feat_names,
    "feature_label": [code_map.get(f, f) for f in all_feat_names],
    "mean_abs_shap": imp_avg
}).sort_values("mean_abs_shap", ascending=False)

# Save CSV
csv_path = "results/shap_mean_abs_importance.csv"
imp_df.to_csv(csv_path, index=False)
print(f"\n✅ Saved SHAP importances to {csv_path}")

print("\nTop 20 features (friendly labels):")
print(imp_df.head(20).to_string(index=False))

# ========================================
# 4. Bar Plot Top 15
# ========================================
top_n = 15
top_features = imp_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature_label"], top_features["mean_abs_shap"], color="skyblue")
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP Value|")
plt.title(f"Top {top_n} Features by SHAP Importance (5-fold avg)")
plt.tight_layout()
plt.savefig("results/shap_importance_plot.png", dpi=300)
print("\n📊 Saved SHAP importance plot to results/shap_importance_plot.png")
plt.show()

# ========================================
# 5. Beeswarm Plot
# ========================================
clf = models[0]
pre = clf.named_steps["pre"]
model = clf.named_steps["clf"]

X_trans = pre.transform(X)
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X_trans)

shap.summary_plot(
    shap_vals, X_trans, feature_names=[code_map.get(f, f) for f in all_feat_names],
    show=False, plot_type="dot", max_display=25
)
plt.savefig("results/shap_beeswarm.png", dpi=300, bbox_inches="tight")
print("📊 Saved SHAP beeswarm plot to results/shap_beeswarm.png")
plt.close()

# ========================================
# 6. Permutation Importance (compute once)
# ========================================
from sklearn.inspection import permutation_importance

res = permutation_importance(
    model, X_trans, y,
    n_repeats=10, random_state=42, scoring="roc_auc"
)

perm_df = pd.DataFrame({
    "feature": [code_map.get(f, f) for f in all_feat_names],
    "perm_mean": res.importances_mean
}).sort_values("perm_mean", ascending=False)

perm_df.to_csv("results/permutation_importance.csv", index=False)
print("✅ Saved permutation importance to results/permutation_importance.csv")
print("\nTop 10 permutation importance features:")
print(perm_df.head(10).to_string(index=False))

# ========================================
# 7. SHAP Report: Bar, Beeswarm, Dependence + Permutation → Combined PDF
# ========================================
from matplotlib.backends.backend_pdf import PdfPages

top_n = 15
top_features_raw = imp_df.head(top_n)["feature_raw"].tolist()

out_dir = "results/shap_dependence"
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(out_dir, "shap_dependence_report.pdf")

with PdfPages(pdf_path) as pdf:
    # --- Page 1: Bar Plot ---
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature_label"], top_features["mean_abs_shap"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP Value|")
    plt.title(f"Top {top_n} Features by SHAP Importance (5-fold avg)")
    plt.tight_layout()
    plt.savefig("results/shap_importance_plot.png", dpi=300)
    pdf.savefig(plt.gcf(), dpi=300, bbox_inches="tight")
    plt.close()
    print("📊 Added Bar Plot to PDF")

    # --- Page 2: Beeswarm Plot ---
    shap.summary_plot(
        shap_vals, X_trans,
        feature_names=[code_map.get(f, f) for f in all_feat_names],
        show=False, plot_type="dot", max_display=25
    )
    plt.title("SHAP Beeswarm Plot (Top 25 Features)")
    plt.savefig("results/shap_beeswarm.png", dpi=300, bbox_inches="tight")
    pdf.savefig(plt.gcf(), dpi=300, bbox_inches="tight")
    plt.close()
    print("🐝 Added Beeswarm Plot to PDF")

    # --- Pages 3+: Dependence Plots for Top-N ---
    for f_raw in top_features_raw:
        if f_raw not in all_feat_names:
            print(f"⚠️ Skipping {f_raw} (not found in transformed features)")
            continue

        label = code_map.get(f_raw, f_raw)
        safe_label = label.replace(":", "").replace("/", "_").replace(" ", "_").replace("–", "-")

        shap.dependence_plot(
            f_raw, shap_vals, X_trans,
            feature_names=all_feat_names,
            show=False
        )
        plt.title(f"Dependence Plot: {label}")

        out_path = os.path.join(out_dir, f"shap_dependence_{safe_label}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

        pdf.savefig(plt.gcf(), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📈 Added Dependence Plot for {label} -> {out_path}")

    # --- Last Page: Permutation Importance ---
    plt.figure(figsize=(10, 6))
    plt.barh(perm_df.head(15)["feature"], perm_df.head(15)["perm_mean"], color="lightcoral")
    plt.gca().invert_yaxis()
    plt.xlabel("Mean Decrease in AUC (Permutation Importance)")
    plt.title("Top 15 Features by Permutation Importance")
    plt.tight_layout()
    plt.savefig("results/permutation_importance_plot.png", dpi=300)
    pdf.savefig(plt.gcf(), dpi=300, bbox_inches="tight")
    plt.close()
    print("🔄 Added Permutation Importance to PDF")

print(f"\n📑 SHAP + Permutation Report saved: {pdf_path}")
# ========================================
# 7. Permutation Importance (sanity check)
# ========================================
res = permutation_importance(model, X_trans, y, n_repeats=10, random_state=42, scoring="roc_auc")
perm_df = pd.DataFrame({
    "feature": [code_map.get(f, f) for f in all_feat_names],
    "perm_mean": res.importances_mean
}).sort_values("perm_mean", ascending=False)

perm_df.to_csv("results/permutation_importance.csv", index=False)
print("✅ Saved permutation importance to results/permutation_importance.csv")
print("\nTop 10 permutation importance features:")
print(perm_df.head(10).to_string(index=False))

# ----------------------------------------
# Regenerate specific problematic plots (debug/strong overlay)
# ----------------------------------------
debug_features = [
    "Status_of_existing_checking_account_A14",
    # add others here if you want
]

for dbg in debug_features:
    if dbg not in all_feat_names:
        print(f"DBG: {dbg} not in feature list, skipping")
        continue
    label = code_map.get(dbg, dbg)
    safe_label = label.replace(":", "").replace("/", "_").replace(" ", "_").replace("–", "-")

    idx = all_feat_names.index(dbg)
    x = X_trans[:, idx]
    y_shap = shap_vals[:, idx]

    # color by interaction if possible, else |shap|
    try:
        inter = explainer.shap_interaction_values(X_trans)
        if isinstance(inter, list):
            inter = inter[0]
        mean_inter = np.mean(np.abs(inter[:, idx, :]), axis=0)
        mean_inter[idx] = 0.0
        interact_idx = int(np.argmax(mean_inter))
        color = X_trans[:, interact_idx]
        color_label = code_map.get(all_feat_names[interact_idx], all_feat_names[interact_idx])
    except Exception:
        color = np.abs(y_shap)
        color_label = "|SHAP value|"

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y_shap, c=color, cmap="viridis", s=20, alpha=0.9)
    cbar = plt.colorbar(sc)
    try:
        cbar.ax.yaxis.set_label_text(color_label)
    except Exception:
        cbar.set_label(color_label)

    ax = plt.gca()
    ax.xaxis.set_label_text(label)
    ax.yaxis.set_label_text(f"SHAP value for {label}")

    # Add a strong white overlay box at the top center to ensure friendly label is visible
    fig = plt.gcf()
    fig.text(0.5, 0.985, label, ha="center", va="top", fontsize=14, fontweight="bold",
             bbox={"facecolor": "white", "alpha": 0.95, "edgecolor": "black"})

    out_png = os.path.join(out_dir, f"shap_dependence_{safe_label}_debug.png")
    out_pdf = os.path.join(out_dir, f"shap_dependence_{safe_label}_debug.pdf")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"DBG: regenerated debug plots -> {out_png} and {out_pdf}")