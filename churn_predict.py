
from pathlib import Path
import pandas as pd

# Project paths
BASE = Path(__file__).resolve().parent
CLEAN = BASE / "data" / "clean"

# Load customer features
df = pd.read_csv(CLEAN / "customer_features.csv")

# Assign churn risk based on days since last completed order
def assign_churn_risk(days):
    if days >= 60:
        return "High"
    elif days >= 30:
        return "Medium"
    else:
        return "Low"

df["churn_risk"] = df["days_since_last_order"].apply(
    assign_churn_risk
)

# Save churn results
output_file = CLEAN / "churn_predictions.csv"
df.to_csv(output_file, index=False)

# Display summary
print("Churn risk analysis completed!")
print("\nCustomer churn risk summary:")

print(
    df["churn_risk"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
)

print("\nSaved file:", output_file)
