"""
E-Commerce Return Insight
train_model.py — Data cleaning, feature engineering, model training, and evaluation.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

def data_path(filename):
    return os.path.join(DATA_DIR, filename)

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  E-Commerce Return Insight — Model Training")
print("=" * 55)

print("\n[1/5] Loading CSV files...")
orders    = pd.read_csv(data_path("orders.csv"))
products  = pd.read_csv(data_path("products.csv"))
customers = pd.read_csv(data_path("customers.csv"))
delivery  = pd.read_csv(data_path("delivery.csv"))
returns   = pd.read_csv(data_path("returns.csv"))

print(f"  orders    : {len(orders):>4} rows")
print(f"  products  : {len(products):>4} rows")
print(f"  customers : {len(customers):>4} rows")
print(f"  delivery  : {len(delivery):>4} rows")
print(f"  returns   : {len(returns):>4} rows")

# ─────────────────────────────────────────────
# 3. CLEAN DATA
# ─────────────────────────────────────────────
print("\n[2/5] Cleaning data...")

# Remove duplicates
orders.drop_duplicates(subset="order_id", inplace=True)
delivery.drop_duplicates(subset="order_id", inplace=True)
products.drop_duplicates(subset="product_id", inplace=True)
customers.drop_duplicates(subset="customer_id", inplace=True)
returns.drop_duplicates(subset="order_id", inplace=True)

# Parse dates
for df, cols in [
    (orders,   ["order_date"]),
    (delivery, ["promised_date", "delivered_date"]),
    (returns,  ["return_date"]),
]:
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Drop rows with invalid order or delivery dates (critical columns)
before = len(orders)
orders.dropna(subset=["order_date"], inplace=True)
delivery.dropna(subset=["promised_date", "delivered_date"], inplace=True)
print(f"  Dropped {before - len(orders)} orders with invalid order_date")

# Remove invalid numeric values
orders = orders[orders["price"] > 0]
orders = orders[orders["quantity"] > 0]
orders["discount"] = orders["discount"].clip(lower=0, upper=100)

# ─────────────────────────────────────────────
# 4. MERGE DATASETS
# ─────────────────────────────────────────────
print("\n[3/5] Merging datasets...")

df = orders.merge(delivery,  on="order_id",   how="inner")
df = df.merge(products,      on="product_id", how="left")
df = df.merge(customers,     on="customer_id", how="left")

# Build return_flag: 1 if order appears in returns, 0 otherwise
return_ids = returns[["order_id", "return_flag"]].copy()
df = df.merge(return_ids, on="order_id", how="left")
df["return_flag"] = df["return_flag"].fillna(0).astype(int)

print(f"  Merged dataset shape: {df.shape}")
print(f"  Return rate: {df['return_flag'].mean():.1%}  "
      f"({df['return_flag'].sum()} returned / {len(df)} total)")

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[4/5] Engineering features...")

# Delivery delay (positive = late, negative = early)
df["delivery_delay_days"] = (
    df["delivered_date"] - df["promised_date"]
).dt.days

# Order value after discount
df["order_value"] = df["quantity"] * df["price"] * (1 - df["discount"] / 100)

# Order month and day-of-week
df["order_month"]   = df["order_date"].dt.month
df["order_dayofweek"] = df["order_date"].dt.dayofweek  # 0=Mon … 6=Sun

# Drop rows where delivery delay is still NaN
df.dropna(subset=["delivery_delay_days"], inplace=True)

# Fill remaining NaN in categorical columns
for col in ["category", "brand", "size", "seller_id", "region",
            "customer_segment", "courier"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Fill order_history NaN
df["order_history"] = df["order_history"].fillna(df["order_history"].median())

# ─────────────────────────────────────────────
# 6. SELECT FEATURES
# ─────────────────────────────────────────────
CATEGORICAL_COLS = [
    col for col in ["category", "brand", "size", "seller_id",
                    "region", "customer_segment", "courier"]
    if col in df.columns
]

NUMERIC_COLS = [
    col for col in ["quantity", "price", "discount",
                    "delivery_delay_days", "order_value",
                    "order_month", "order_dayofweek", "order_history"]
    if col in df.columns
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

print(f"  Numeric features  : {NUMERIC_COLS}")
print(f"  Categorical features: {CATEGORICAL_COLS}")

# Encode categoricals
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

ENCODED_COLS = NUMERIC_COLS + [c + "_enc" for c in CATEGORICAL_COLS]

X = df[ENCODED_COLS].copy()
y = df["return_flag"].copy()

# ─────────────────────────────────────────────
# 7. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────
# 8. TRAIN MODEL
# ─────────────────────────────────────────────
print("\n[5/5] Training Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 9. EVALUATE
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
cm        = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 55)
print("  Model Evaluation Results")
print("=" * 55)
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"                 Predicted")
print(f"                 No   Yes")
print(f"  Actual No   [{cm[0][0]:>4}  {cm[0][1]:>4}]")
print(f"  Actual Yes  [{cm[1][0]:>4}  {cm[1][1]:>4}]")
print("\n  Full Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Not Returned", "Returned"]))

# Feature importance
importances = pd.Series(
    model.feature_importances_, index=ENCODED_COLS
).sort_values(ascending=False)
print("  Top 10 Feature Importances:")
for feat, imp in importances.head(10).items():
    bar = "█" * int(imp * 50)
    print(f"    {feat:<30} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────
# 10. SAVE MODEL BUNDLE
# ─────────────────────────────────────────────
model_bundle = {
    "model": model,
    "label_encoders": label_encoders,
    "categorical_cols": CATEGORICAL_COLS,
    "numeric_cols": NUMERIC_COLS,
    "encoded_cols": ENCODED_COLS,
    "feature_cols": FEATURE_COLS,
    "metrics": {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "confusion_matrix": cm.tolist(),
    },
    # Store unique values so predict.py can show valid choices
    "category_values": {
        col: sorted(df[col].unique().tolist())
        for col in CATEGORICAL_COLS
    },
    # Store numeric feature stats for validation
    "numeric_stats": {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
        }
        for col in NUMERIC_COLS
    },
}

model_path = os.path.join(BASE_DIR, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model_bundle, f)

print(f"\n  Model saved → {model_path}")
print("\n" + "=" * 55)
print("  Training complete. Run: python predict.py")
print("=" * 55)
