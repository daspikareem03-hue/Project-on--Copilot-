
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

# ============================================================
# PART A: SWIGGY INSTAMART DATA CLEANING AND SQLITE DATABASE
# ============================================================

# 1. Define project folders
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

CLEAN.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 2. Load raw datasets
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")
tickets = pd.read_csv(
    RAW / "tickets.csv",
    keep_default_na=False
)

print("\n========== RAW DATA AUDIT ==========")

def audit_data(name, df):
    print(f"\n{name}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isna().sum())

    blank_counts = (
        df.select_dtypes(include="object")
        .apply(lambda col: col.str.strip().eq("").sum())
    )

    print("Blank strings:\n", blank_counts)
    print("Exact duplicate rows:", df.duplicated().sum())


audit_data("Customers", customers)
audit_data("Products", products)
audit_data("Orders", orders)
audit_data("Tickets", tickets)


# ============================================================
# 3. CLEAN CUSTOMERS
# ============================================================

print("\n========== CLEANING CUSTOMERS ==========")

# Remove exact duplicate rows
customers = customers.drop_duplicates().copy()

# Standardize city names
customers["city"] = (
    customers["city"]
    .astype("string")
    .str.strip()
    .str.title()
)

# Standardize gender
customers["gender"] = (
    customers["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
)

customers["gender"] = customers["gender"].replace({
    "m": "male",
    "man": "male",
    "male": "male",
    "f": "female",
    "woman": "female",
    "female": "female"
})

customers["gender"] = customers["gender"].fillna("unknown")
customers.loc[
    customers["gender"].eq(""),
    "gender"
] = "unknown"

# Convert age to numeric
customers["age"] = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

# Invalid ages become missing
customers.loc[
    ~customers["age"].between(10, 100),
    "age"
] = np.nan

# Fill missing ages using city median
city_medians = customers.groupby("city")["age"].transform("median")

customers["age"] = customers["age"].fillna(city_medians)

# Fallback if any age remains missing
customers["age"] = customers["age"].fillna(
    customers["age"].median()
)

# Convert signup date
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

print("Clean customers:", customers.shape)
print("Missing customer ages:", customers["age"].isna().sum())


# ============================================================
# 4. CLEAN PRODUCTS
# ============================================================

print("\n========== CLEANING PRODUCTS ==========")

products = products.drop_duplicates().copy()

products["category"] = (
    products["category"]
    .astype("string")
    .str.strip()
    .str.title()
)

products["price_inr"] = pd.to_numeric(
    products["price_inr"],
    errors="coerce"
)

# Remove products with missing or invalid prices
products = products.loc[
    products["price_inr"].notna()
    & products["price_inr"].ge(0)
].copy()

print("Clean products:", products.shape)


# ============================================================
# 5. CLEAN ORDERS
# ============================================================

print("\n========== CLEANING ORDERS ==========")

# Convert order timestamp
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)

# Standardize status
orders["status"] = (
    orders["status"]
    .astype("string")
    .str.strip()
    .str.lower()
)

# Remove exact duplicate rows
orders = orders.drop_duplicates().copy()

# Keep the latest record for each order_id
orders = (
    orders.sort_values("order_ts")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

# Convert numeric columns
orders["quantity"] = pd.to_numeric(
    orders["quantity"],
    errors="coerce"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"],
    errors="coerce"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# Negative delivery times are invalid
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = np.nan

# Join product details
orders = orders.merge(
    products[["product_id", "category", "price_inr"]],
    on="product_id",
    how="left",
    suffixes=("", "_product")
)

# Fill missing order amounts using product price,
# quantity, and discount percentage
orders["discount_pct"] = orders["discount_pct"].fillna(0)
orders["quantity"] = orders["quantity"].fillna(1)

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

# Remove temporary product price column
orders = orders.drop(columns=["price_inr"])

# Fill missing delivery times for completed orders
completed_mask = orders["status"].eq("completed")

category_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .median()
)

missing_completed = (
    completed_mask
    & orders["delivery_minutes"].isna()
)

orders.loc[missing_completed, "delivery_minutes"] = (
    orders.loc[missing_completed, "category"]
    .map(category_medians)
)

# Fallback to overall completed-order median
completed_median = orders.loc[
    completed_mask, "delivery_minutes"
].median()

orders.loc[
    completed_mask & orders["delivery_minutes"].isna(),
    "delivery_minutes"
] = completed_median

# Cancelled orders should have missing delivery time
orders.loc[
    orders["status"].eq("cancelled"),
    "delivery_minutes"
] = np.nan

# Remove temporary category column
orders = orders.drop(columns=["category"])

print("Clean orders:", orders.shape)
print("Unique order IDs:", orders["order_id"].nunique())
print("Order status counts:\n", orders["status"].value_counts())
print(
    "Missing completed delivery times:",
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)


# ============================================================
# 6. SAVE CLEANED CSV FILES
# ============================================================

print("\n========== SAVING CLEANED CSV FILES ==========")

customers.to_csv(CLEAN / "customers_clean.csv", index=False)
products.to_csv(CLEAN / "products_clean.csv", index=False)
orders.to_csv(CLEAN / "orders_clean.csv", index=False)

print("Saved customers_clean.csv")
print("Saved products_clean.csv")
print("Saved orders_clean.csv")


# ============================================================
# 7. CREATE SQLITE DATABASE
# ============================================================

print("\n========== CREATING SQLITE DATABASE ==========")

# Store timestamps as readable text
customers["signup_date"] = customers["signup_date"].dt.strftime(
    "%Y-%m-%d"
)

orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

with sqlite3.connect(DB_PATH) as con:

    customers.to_sql(
        "customers",
        con,
        if_exists="replace",
        index=False
    )

    products.to_sql(
        "products",
        con,
        if_exists="replace",
        index=False
    )

    orders.to_sql(
        "orders",
        con,
        if_exists="replace",
        index=False
    )

    # Verify database tables
    tables = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """,
        con
    )

    print("Database tables:")
    print(tables)

    # Verify table row counts
    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS total_rows FROM {table}",
            con
        )

        print(f"{table}: {count.iloc[0, 0]} rows")


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print("\n========== PART A COMPLETED ==========")

print("Cleaned CSV files location:", CLEAN)
print("SQLite database location:", DB_PATH)

print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))

print(
    "Completed order revenue (INR):",
    round(
        orders.loc[
            orders["status"].eq("completed"),
            "order_amount"
        ].sum(),
        2
    )
)

print("\nPART A DATA CLEANING AND DATABASE CREATION COMPLETED!")
print("RAW folder:", RAW)
print("Customers file exists:", (RAW / "customers.csv").exists())