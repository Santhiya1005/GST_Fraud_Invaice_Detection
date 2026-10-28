import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

INPUT_FILE = "engineered_invoices.csv"
OUTPUT_FILE = "fraud_results.csv"
MODEL_FILE = "isolation_forest_model.joblib"

print("Loading engineered dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# =========================================================
# 1. ML FEATURES
# =========================================================

FEATURES = [
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

X = df[FEATURES].astype(np.float32)

print("ML matrix:", X.shape)

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    max_samples=10000,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X)

print("Isolation Forest training complete.")

df["if_prediction"] = model.predict(X)

# -1 = anomaly
#  1 = normal

df["is_anomaly"] = (
    df["if_prediction"] == -1
).astype(int)

# Higher value = more suspicious
raw_anomaly = -model.decision_function(X)

# Normalize to 0-100 using percentile ranking
df["isolation_score"] = (
    pd.Series(raw_anomaly)
    .rank(pct=True) * 100
).astype(np.float32)


print("Calculating WFRS...")

wfrs = np.zeros(len(df), dtype=np.float32)

# Tax calculation mismatch
wfrs += np.where(
    df["tax_difference"] > 0.01,
    30,
    0
)

# Total invoice calculation mismatch
wfrs += np.where(
    df["total_error_ratio"] > 0.01,
    25,
    0
)

# Large discount
wfrs += np.where(
    df["high_discount_flag"] == 1,
    10,
    0
)

# Unusual tax percentage
wfrs += np.where(
    df["high_tax_flag"] == 1,
    10,
    0
)

# Negative values
wfrs += np.where(
    df["negative_amount_flag"] == 1,
    15,
    0
)

# Payment delay
wfrs += np.where(
    df["payment_after_due_flag"] == 1,
    5,
    0
)

# Duplicate invoice number
wfrs += np.where(
    df["invoice_number_frequency"] > 1,
    5,
    0
)

df["wfrs"] = np.clip(wfrs, 0, 100)


# 60% ML anomaly score
# 40% GST/business rule score

df["fraud_risk_score"] = (
    0.60 * df["isolation_score"] +
    0.40 * df["wfrs"]
)

df["fraud_risk_score"] = (
    df["fraud_risk_score"]
    .clip(0, 100)
    .round(2)
)


def risk_category(score):

    if score >= 75:
        return "CRITICAL"

    elif score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    else:
        return "LOW"


df["risk_category"] = (
    df["fraud_risk_score"]
    .apply(risk_category)
)


def explain_row(row):

    reasons = []

    if row["tax_difference"] > 0.01:
        reasons.append("Tax calculation mismatch")

    if row["total_error_ratio"] > 0.01:
        reasons.append("Invoice total mismatch")

    if row["high_discount_flag"] == 1:
        reasons.append("High discount")

    if row["high_tax_flag"] == 1:
        reasons.append("Unusual tax percentage")

    if row["negative_amount_flag"] == 1:
        reasons.append("Negative financial value")

    if row["payment_after_due_flag"] == 1:
        reasons.append("Payment after due date")

    if row["invoice_number_frequency"] > 1:
        reasons.append("Duplicate invoice number")

    if row["is_anomaly"] == 1:
        reasons.append("Isolation Forest anomaly")

    if not reasons:
        reasons.append("No major risk indicator")

    return "; ".join(reasons)


print("Generating explanations...")

df["risk_reason"] = df.apply(
    explain_row,
    axis=1
)

RESULT_COLUMNS = [
    "is_anomaly",
    "isolation_score",
    "wfrs",
    "fraud_risk_score",
    "risk_category",
    "risk_reason"
]

df[RESULT_COLUMNS].to_csv(
    OUTPUT_FILE,
    index=False
)

joblib.dump(model, MODEL_FILE)

print("\n========================================")
print("FRAUD DETECTION COMPLETE")
print("========================================")

print("\nRisk Distribution:")
print(
    df["risk_category"]
    .value_counts()
)

print("\nAnomalies:")
print(
    df["is_anomaly"].sum()
)

print("\nAverage Risk Score:")
print(
    round(df["fraud_risk_score"].mean(), 2)
)

print("\nOutput file:", OUTPUT_FILE)
print("Model file:", MODEL_FILE)