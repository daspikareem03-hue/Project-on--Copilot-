
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

rfm = pd.read_csv(OUTPUTS / "rfm.csv")

features = rfm[["recency", "frequency", "monetary"]]

# Standardize RFM features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Final K-Means model: K = 4
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(scaled_features)

# Profile each segment
profile = rfm.groupby("segment_id").agg(
    customers=("customer_id", "count"),
    avg_recency=("recency", "mean"),
    avg_frequency=("frequency", "mean"),
    avg_monetary=("monetary", "mean")
).round(2)

print("\nSegment Profile:")
print(profile)

# Name segments based on their RFM profile
monetary_rank = profile["avg_monetary"].rank(
    ascending=False,
    method="min"
)

names = {
    1: "Premium",
    2: "Regular",
    3: "Occasional",
    4: "At Risk"
}

# Highest monetary segment gets Premium
premium_segment = profile["avg_monetary"].idxmax()

segment_names = {}
for segment_id in profile.index:
    if segment_id == premium_segment:
        segment_names[segment_id] = "Premium"
    else:
        segment_names[segment_id] = f"Segment {segment_id}"

rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"] == premium_segment

# PCA for 2D visualization features
pca = PCA(n_components=2, random_state=42)
pca_features = pca.fit_transform(scaled_features)

rfm["pca1"] = pca_features[:, 0]
rfm["pca2"] = pca_features[:, 1]

print("\nPCA Explained Variance Ratios:")
print(pca.explained_variance_ratio_)
print("Total:", pca.explained_variance_ratio_.sum())

# Save required output
final_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
]

rfm[final_columns].to_csv(
    OUTPUTS / "customer_segments.csv",
    index=False
)

print("\nCustomer segmentation completed!")
print("Premium segment ID:", premium_segment)
print("Saved:", OUTPUTS / "customer_segments.csv")
