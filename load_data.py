import pandas as pd

# Define column names based on german.doc (UCI documentation)
columns = [
    "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
    "Savings", "Employment", "InstallmentRate", "PersonalStatusSex",
    "OtherDebtors", "ResidenceDuration", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "ExistingCredits",
    "Job", "LiableDependents", "Telephone", "ForeignWorker", "Target"
]

def load_german_credit():
    # Load dataset
    df = pd.read_csv("german.data", 
                     sep="\s+",  
                     header=None, 
                     names=columns)
    return df

if __name__ == "__main__":
    data = load_german_credit()
    print(data.head())   # preview first 5 rows
    print("\nShape of dataset:", data.shape)