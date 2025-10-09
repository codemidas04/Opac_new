import pandas as pd

# Define column names (consistent with EDA + preprocessing.py)
columns = [
    "Status_of_existing_checking_account", "Duration_in_month", "Credit_history",
    "Purpose", "Credit_amount", "Savings_account/bonds",
    "Present_employment_since", "Installment_rate_in_percentage_of_disposable_income",
    "Personal_status_and_sex", "Other_debtors/guarantors",
    "Present_residence_since", "Property", "Age_in_years",
    "Other_installment_plans", "Housing", "Number_of_existing_credits_at_this_bank",
    "Job", "Number_of_people_being_liable_to_provide_maintenance_for",
    "Telephone", "Foreign_worker", "Credit_risk"
]

def load_german_credit():
    df = pd.read_csv(
        "german.data",
        sep="\s+",
        header=None,
        names=columns
    )
    # Add standardized target column
    df["target"] = df["Credit_risk"].map({1: 0, 2: 1})
    return df

if __name__ == "__main__":
    data = load_german_credit()
    print(data.head())
    print("\nShape of dataset:", data.shape)