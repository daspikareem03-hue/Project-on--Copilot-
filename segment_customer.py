
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# File paths
BASE = Path(__file__).resolve().parent
CLEAN = BASE / "data" / "clean"

# Load customer features
df = pd.read_csv(CLEAN / "customer_features.csv")

# Features used for segmentation
df["no_completed_orders"] = (df["total_orders"] == 0).astype(int)

# Cap recency to reduce the effect of placeholder value 999
df["recency_days"] = df["days_since_last_order"].clip(upper=90)

feature_columns = [
    "total_orders",
    "total_spending",
    "average_order_value",
    "recency_days",
    "no_completed_orders"
]

X = df[feature_columns]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create 3 customer groups
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["customer_segment"] = model.fit_predict(X_scaled)

# Show segment summary
summary = df.groupby("customer_segment").agg(
    customers=("customer_id", "count"),
    average_orders=("total_orders", "mean"),
    average_spending=("total_spending", "mean"),
    average_recency=("recency_days", "mean")
).round(2)

print("Customer segmentation completed!")
print("\nSegment summary:")
print(summary)

# Save results
output_file = CLEAN / "customer_segments.csv"
df.to_csv(output_file, index=False)

print("\nSaved file:", output_file)
