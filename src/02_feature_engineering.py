import pandas as pd
import numpy as np

INPUT_FILE = "test_invoices.csv"
OUTPUT_FILE = "engineered_invoices.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# =========================================================
# 1. DATE FEATURES
# =========================================================

df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")

df["invoice_year"] = df["invoice_date"].dt.year
df["invoice_month"] = df["invoice_date"].dt.month
df["invoice_day"] = df["invoice_date"].dt.day
df["invoice_weekday"] = df["invoice_date"].dt.weekday

# Days allowed for payment
df["payment_terms_days"] = (
    df["due_date"] - df["invoice_date"]
).dt.days

# Actual payment delay
df["payment_delay_days"] = (
    df["payment_date"] - df["due_date"]
).dt.days

df["payment_delay_days"] = df["payment_delay_days"].fillna(0)

# =========================================================
# 2. INVOICE AMOUNT FEATURES
# =========================================================

df["calculated_subtotal"] = (
    df["quantity"] * df["unit_price"]
)

df["calculated_discount"] = (
    df["calculated_subtotal"] *
    df["discount_percent"] / 100
)

df["calculated_tax"] = (
    (df["calculated_subtotal"] - df["discount_amount"]) *
    df["tax_percent"] / 100
)

df["calculated_total"] = (
    df["subtotal"]
    - df["discount_amount"]
    + df["tax_amount"]
    + df["shipping_cost"]
)

# =========================================================
# 3. ARITHMETIC CONSISTENCY FEATURES
# =========================================================

df["subtotal_difference"] = (
    df["subtotal"] - df["calculated_subtotal"]
).abs()

df["discount_difference"] = (
    df["discount_amount"] - df["calculated_discount"]
).abs()

df["tax_difference"] = (
    df["tax_amount"] - df["calculated_tax"]
).abs()

df["total_difference"] = (
    df["total_amount"] - df["calculated_total"]
).abs()

# Relative total inconsistency
df["total_error_ratio"] = (
    df["total_difference"] /
    df["total_amount"].abs().replace(0, np.nan)
).fillna(0)

# =========================================================
# 4. BASIC RISK FEATURES
# =========================================================

df["high_discount_flag"] = (
    df["discount_percent"] >= 30
).astype(int)

df["high_tax_flag"] = (
    df["tax_percent"] >= 28
).astype(int)

df["zero_tax_flag"] = (
    df["tax_percent"] == 0
).astype(int)

df["negative_amount_flag"] = (
    (df["quantity"] < 0) |
    (df["unit_price"] < 0) |
    (df["subtotal"] < 0) |
    (df["total_amount"] < 0)
).astype(int)

df["payment_after_due_flag"] = (
    df["payment_delay_days"] > 0
).astype(int)

# =========================================================
# 5. FREQUENCY FEATURES
# =========================================================

print("Creating frequency features...")

df["customer_frequency"] = (
    df.groupby("customer_id")["invoice_id"]
    .transform("count")
)

df["vendor_frequency"] = (
    df.groupby("vendor_id")["invoice_id"]
    .transform("count")
)

df["product_frequency"] = (
    df.groupby("product_id")["invoice_id"]
    .transform("count")
)

df["invoice_number_frequency"] = (
    df.groupby("invoice_number")["invoice_id"]
    .transform("count")
)

# =========================================================
# 6. CLEAN INVALID VALUES
# =========================================================

numeric_columns = [
    "quantity",
    "unit_price",
    "discount_percent",
    "tax_percent",
    "subtotal",
    "discount_amount",
    "tax_amount",
    "shipping_cost",
    "total_amount"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill numeric missing values with median
for col in df.select_dtypes(include=["number"]).columns:
    df[col] = df[col].fillna(df[col].median())

# =========================================================
# 7. SELECT FINAL ML FEATURES
# =========================================================

ML_FEATURES = [
    "quantity",
    "unit_price",
    "discount_percent",
    "tax_percent",
    "subtotal",
    "discount_amount",
    "tax_amount",
    "shipping_cost",
    "total_amount",

    "payment_terms_days",
    "payment_delay_days",

    "subtotal_difference",
    "discount_difference",
    "tax_difference",
    "total_difference",
    "total_error_ratio",

    "high_discount_flag",
    "high_tax_flag",
    "zero_tax_flag",
    "negative_amount_flag",
    "payment_after_due_flag",

    "customer_frequency",
    "vendor_frequency",
    "product_frequency",
    "invoice_number_frequency"
]

print("\nFinal ML features:")
for feature in ML_FEATURES:
    print("-", feature)

# =========================================================
# 8. SAVE ENGINEERED DATA
# =========================================================

df[ML_FEATURES].to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n===================================")
print("FEATURE ENGINEERING COMPLETE")
print("===================================")
print("Output:", OUTPUT_FILE)
print("Shape:", df[ML_FEATURES].shape)
