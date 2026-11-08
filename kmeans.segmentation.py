
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

rfm = pd.read_csv(OUTPUTS / "rfm.csv")

features = rfm[["recency", "frequency", "monetary"]]

# Standardize RFM features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

results = []

# Test K values from 2 to 8
for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(scaled_features)

    inertia = model.inertia_
    silhouette = silhouette_score(scaled_features, labels)

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette": silhouette
    })

results_df = pd.DataFrame(results)

print("\nK-Means Evaluation Results:")
print(results_df.round(3).to_string(index=False))

results_df.to_csv(OUTPUTS / "kmeans_evaluation.csv", index=False)

print("\nSaved file:", OUTPUTS / "kmeans_evaluation.csv")
