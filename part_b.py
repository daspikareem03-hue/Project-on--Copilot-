
from pathlib import Path
import pandas as pd

# Project paths
# BASE = Path(__file__).resolve().parent
CLEAN = BASE / "data" / "clean"

# Load cleaned data
customers = pd.read_csv(CLEAN / "customers_clean.csv")
orders = pd.read_csv(CLEAN / "orders_clean.csv")

# Keep completed orders
completed = orders[orders["status"] == "completed"].copy()

# Convert order date
completed["order_ts"] = pd.to_datetime(completed["order_ts"])

# Create customer features
features = completed.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    total_spending=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    last_order_date=("order_ts", "max")
).reset_index()

# Calculate days since last order
reference_date = completed["order_ts"].max()

features["days_since_last_order"] = (
    reference_date - features["last_order_date"]
).dt.days

# Add customer information
features = features.merge(
    customers,
    on="customer_id",
    how="right"
)

# Fill missing values for customers with no completed orders
features["total_orders"] = features["total_orders"].fillna(0)
features["total_spending"] = features["total_spending"].fillna(0)
features["average_order_value"] = features["average_order_value"].fillna(0)
features["days_since_last_order"] = (
    features["days_since_last_order"].fillna(999)
)

# Save output
output_file = CLEAN / "customer_features.csv"
features.to_csv(output_file, index=False)

print("Customer features created successfully!")
print("Total customers:", len(features))
print("Saved file:", output_file)
print(features.head())
