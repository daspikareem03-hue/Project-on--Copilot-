
from pathlib import Path
import pandas as pd

# Project folders
ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)

# Load Part A cleaned data
orders = pd.read_csv(CLEAN / "orders_clean.csv")
customers = pd.read_csv(CLEAN / "customers_clean.csv")

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Assignment snapshot date
snapshot = pd.Timestamp("2025-10-01 00:00:00")

# Use completed orders before the snapshot
completed = orders[
    (orders["status"].str.lower() == "completed")
    & (orders["order_ts"] < snapshot)
].copy()

# Build RFM table
rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Recency = whole days since last completed order
rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days

# Keep required RFM columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Save output
rfm.to_csv(OUTPUTS / "rfm.csv", index=False)

# Print reference checks
print("\nRFM Analysis Completed!")
print("Total customers:", len(rfm))

print("\nRFM Summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())

print("\nReference values from assignment:")
print("Customers: 1175")
print("Recency median: 19")
print("Recency minimum: 0")
print("Recency maximum: 180")
print("Frequency mean: 9.46")
print("Frequency maximum: 33")
print("Monetary mean: 2552.06")
print("Monetary maximum: 14766.55")

print("\nSaved file:", OUTPUTS / "rfm.csv")

