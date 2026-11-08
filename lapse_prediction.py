
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"

OUTPUTS.mkdir(parents=True, exist_ok=True)

# Load Part A cleaned data
orders = pd.read_csv(CLEAN / "orders_clean.csv")
customers = pd.read_csv(CLEAN / "customers_clean.csv")

# Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Assignment cutoff dates
cutoff = pd.Timestamp("2025-09-01 00:00:00")
lapse_end = pd.Timestamp("2025-10-01 00:00:00")

# Normalize order status
orders["status"] = orders["status"].str.lower().str.strip()

# Use only orders before September 1 for features
historical = orders[orders["order_ts"] < cutoff].copy()

# Completed orders before cutoff
completed_before = historical[
    historical["status"] == "completed"
].copy()

# Customers with at least one completed order before cutoff
cohort_ids = completed_before["customer_id"].unique()

cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

# Lapse label:
# 1 = no completed order during September
# 0 = at least one completed order during September
september_completed = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= cutoff) &
    (orders["order_ts"] < lapse_end)
]

active_in_september = set(september_completed["customer_id"])

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(active_in_september)
).astype(int)

# Build historical RFM-style features
features = completed_before.groupby("customer_id").agg(
    last_completed_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    mean_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

features["recency"] = (
    cutoff - features["last_completed_order"]
).dt.days

# Mean discount across historical orders
discount = historical.groupby("customer_id").agg(
    mean_discount_pct=("discount_pct", "mean")
).reset_index()

# Cancellation rate among all historical orders
status_counts = historical.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    cancelled_orders=(
        "status",
        lambda x: (x == "cancelled").sum()
    )
).reset_index()

status_counts["cancellation_rate"] = (
    status_counts["cancelled_orders"] /
    status_counts["total_orders"].replace(0, np.nan)
)

# Load product categories from the products table 
products = pd.read_csv(CLEAN / "products_clean.csv")
                       
products  = products[["product_id", "category"]].drop_duplicates() 

#join historical orders with product categories
historical_products = historical.merge(
    products,
    on="product_id",
    how="left"
)

category_features = historical_products.groupby("customer_id")[
    "category"
].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
).reset_index(name="most_frequent_category")

# Combine features with customer information
dataset = cohort.merge(features, on="customer_id", how="left")
dataset = dataset.merge(discount, on="customer_id", how="left")
dataset = dataset.merge(
    status_counts[["customer_id", "cancellation_rate"]],
    on="customer_id",
    how="left"
)
dataset = dataset.merge(
    category_features,
    on="customer_id",
    how="left"
)

# Customer tenure as of cutoff
dataset["tenure_days"] = (
    cutoff - dataset["signup_date"]
).dt.days

# Keep useful customer attributes
required_columns = [
    "customer_id",
    "lapsed",
    "recency",
    "frequency",
    "monetary",
    "mean_discount_pct",
    "cancellation_rate",
    "mean_delivery_minutes",
    "most_frequent_category",
    "city",
    "age",
    "gender",
    "tenure_days"
]

dataset = dataset[required_columns]

# Check missing values
print("\nLapse Cohort Summary:")
print("Customers:", len(dataset))
print("Lapsed:", int(dataset["lapsed"].sum()))
print("Not lapsed:", int((dataset["lapsed"] == 0).sum()))
print("Lapse rate:", round(dataset["lapsed"].mean() * 100, 2), "%")

print("\nMissing values:")
print(dataset.isna().sum())

# Save prepared dataset
dataset.to_csv(OUTPUTS / "lapse_features.csv", index=False)

print("\nLapse feature dataset created!")
print("Saved:", OUTPUTS / "lapse_features.csv")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Load prepared lapse dataset
data = pd.read_csv(OUTPUTS / "lapse_features.csv")

# Target
y = data["lapsed"]

# Features: exclude customer ID and target
X = data.drop(columns=["customer_id", "lapsed"])

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "mean_discount_pct",
    "cancellation_rate",
    "mean_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "most_frequent_category",
    "city",
    "gender"
]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# Baseline: predict every customer as not lapsed
baseline_predictions = np.zeros(len(y_test), dtype=int)

print("\nBaseline Results:")
print("Accuracy:", round(accuracy_score(y_test, baseline_predictions), 3))
print("Lapsed F1:", round(
    f1_score(y_test, baseline_predictions, zero_division=0), 3
))

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Logistic Regression pipeline
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_grid = GridSearchCV(
    logistic_pipeline,
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
    n_jobs=-1
)

logistic_grid.fit(X_train, y_train)

print("\nLogistic Regression:")
print("Best parameters:", logistic_grid.best_params_)
print("Best CV F1:", round(logistic_grid.best_score_, 3))

# Random Forest pipeline
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid={
        "model__max_depth": [3, 5, 8],
        "model__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nRandom Forest:")
print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", round(rf_grid.best_score_, 3))

# Evaluate both models on untouched test data
for name, model in [
    ("Logistic Regression", logistic_grid.best_estimator_),
    ("Random Forest", rf_grid.best_estimator_)
]:
    predictions = model.predict(X_test)

    print(f"\n{name} Test Results:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 3))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 3))
    print("F1:", round(f1_score(y_test, predictions, zero_division=0), 3))
    
# ==========================================
# Oversampling Logistic Regression
# ==========================================

# Combine training features and labels
train_data = X_train.copy()
train_data["lapsed"] = y_train

# Separate lapsed and non-lapsed customers
lapsed_rows = train_data[train_data["lapsed"] == 1]
not_lapsed_rows = train_data[train_data["lapsed"] == 0]

print("\nTraining class counts before oversampling:")
print(train_data["lapsed"].value_counts())

# Oversample lapsed customers to match the non-lapsed class
lapsed_oversampled = lapsed_rows.sample(
    n=len(not_lapsed_rows),
    replace=True,
    random_state=42
)

# Combine balanced training data
balanced_train = pd.concat(
    [not_lapsed_rows, lapsed_oversampled],
    ignore_index=True
)

X_balanced = balanced_train.drop(columns=["lapsed"])
y_balanced = balanced_train["lapsed"]

print("\nTraining class counts after oversampling:")
print(y_balanced.value_counts())

# Refit the best Logistic Regression model
oversampled_logistic = logistic_grid.best_estimator_
oversampled_logistic.fit(X_balanced, y_balanced)

# Evaluate on the original, untouched test set
oversampled_predictions = oversampled_logistic.predict(X_test)

print("\nOversampled Logistic Regression Test Results:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, oversampled_predictions))

print("Precision:", round(
    precision_score(y_test, oversampled_predictions, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y_test, oversampled_predictions, zero_division=0), 3
))
print("F1:", round(
    f1_score(y_test, oversampled_predictions, zero_division=0), 3
))

# ==========================================
# Feature Importance and Model Saving
# ==========================================

import joblib

# Random Forest feature importance
best_rf = rf_grid.best_estimator_

rf_feature_names = best_rf.named_steps[
    "preprocessor"
].get_feature_names_out()

rf_importance = pd.DataFrame({
    "feature": rf_feature_names,
    "importance": best_rf.named_steps["model"].feature_importances_
}).sort_values("importance", ascending=False)

print("\nRandom Forest Top 5 Feature Importance:")
print(rf_importance.head(5).to_string(index=False))

# Logistic Regression coefficient signs
lr_feature_names = oversampled_logistic.named_steps[
    "preprocessor"
].get_feature_names_out()

lr_coefficients = oversampled_logistic.named_steps[
    "model"
].coef_[0]

lr_importance = pd.DataFrame({
    "feature": lr_feature_names,
    "coefficient": lr_coefficients
})

lr_importance["direction"] = np.where(
    lr_importance["coefficient"] > 0,
    "Increases lapse risk",
    "Decreases lapse risk"
)

lr_importance["abs_coefficient"] = (
    lr_importance["coefficient"].abs()
)

lr_importance = lr_importance.sort_values(
    "abs_coefficient",
    ascending=False
)

print("\nTop 3 Logistic Regression Feature Signs:")
print(
    lr_importance[
        ["feature", "coefficient", "direction"]
    ].head(3).to_string(index=False)
)

# Save the selected model
model_path = OUTPUTS / "lapse_model.joblib"

joblib.dump(oversampled_logistic, model_path)

print("\nSaved selected model:", model_path)

# Reload the saved model
loaded_model = joblib.load(model_path)

# Check prediction consistency
predictions_before = oversampled_logistic.predict(X_test)
predictions_after = loaded_model.predict(X_test)

print(
    "\nReloaded model predictions match:",
    np.array_equal(predictions_before, predictions_after)
)

# ==========================================
# Premium Customer Retention List
# ==========================================

# Get Premium customers from segmentation output
segments = pd.read_csv(OUTPUTS / "customer_segments.csv")

premium_ids = set(
    segments.loc[segments["is_premium"] == True, "customer_id"]
)

# Recover customer IDs corresponding to the untouched test set
test_customer_ids = data.loc[X_test.index, "customer_id"]

retention = pd.DataFrame({
    "customer_id": test_customer_ids.values,
    "lapse_probability": loaded_model.predict_proba(X_test)[:, 1],
    "actually_lapsed": y_test.values
})

# Keep Premium customers only
retention = retention[
    retention["customer_id"].isin(premium_ids)
]

# Rank by predicted lapse probability
retention = retention.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention.to_csv(
    OUTPUTS / "retention_list.csv",
    index=False
)

print("\nTop 10 Premium Customer Retention List:")
print(retention.to_string(index=False))

print("\nPremium test customers in retention list:", len(retention))
print(
    "Actually lapsed among listed customers:",
    int(retention["actually_lapsed"].sum())
)

print("\nPart B model outputs completed!")