from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from yaml.loader import SafeLoader

import json
import uuid
from supabase import create_client

@st.cache_resource
def get_supabase_client():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SECRET_KEY"],
    )


def save_analysis_results(results_df, batch_id):
    """Save analysis results under the currently logged-in username."""
    username = st.session_state.get("username")

    if not username:
        raise ValueError(
            "Logged-in username was not found. "
            "Cannot safely associate records with a user."
        )

    client = get_supabase_client()

    # Convert pandas/numpy values to JSON-compatible Python values.
    records = json.loads(
        results_df.to_json(orient="records", date_format="iso")
    )

    rows = []

    for record in records:
        rows.append({
            "username": str(username),
            "batch_id": batch_id,
            "invoice_id": str(
                record.get("invoice_id", record.get("invoice_number", ""))
            ),
            "invoice_data": record,
            "risk_score": record.get(
                "fraud_risk_score", record.get("risk_score")
            ),
            "risk_level": record.get(
                "risk_category", record.get("risk_level")
            ),
            "is_anomaly": bool(record.get("is_anomaly", False)),
            "risk_reason": str(
                record.get("risk_reason", record.get("suspicious_reasons", ""))
            ),
        })

    # Insert in smaller chunks for larger Excel files.
    # An empty dataframe should not trigger a database request.
    for start in range(0, len(rows), 500):
        client.table("invoice_analyses").insert(
            rows[start:start + 500]
        ).execute()


def load_saved_analyses(limit=1000):
    """Load the current user's saved analyses from Supabase."""
    username = st.session_state.get("username")
    if not username:
        return pd.DataFrame()

    client = get_supabase_client()
    response = (
        client.table("invoice_analyses")
        .select("invoice_id, invoice_data, risk_score, risk_level, is_anomaly, risk_reason, analyzed_at, batch_id")
        .eq("username", str(username))
        .order("analyzed_at", desc=True)
        .limit(int(limit))
        .execute()
    )

    saved_rows = []
    for item in (response.data or []):
        invoice_data = item.get("invoice_data") or {}
        if not isinstance(invoice_data, dict):
            invoice_data = {}
        # Keep original invoice fields, while ensuring saved analysis columns exist.
        record = dict(invoice_data)
        record.setdefault("invoice_id", item.get("invoice_id"))
        record.setdefault("fraud_risk_score", item.get("risk_score"))
        record.setdefault("risk_category", item.get("risk_level"))
        record.setdefault("is_anomaly", item.get("is_anomaly", False))
        record.setdefault("risk_reason", item.get("risk_reason", ""))
        record.setdefault("analyzed_at", item.get("analyzed_at"))
        record.setdefault("batch_id", item.get("batch_id"))
        saved_rows.append(record)

    return pd.DataFrame(saved_rows)

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="GST Invoice Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
AUTH_FILE = BASE_DIR / "users.yaml"
DATA_FILE = BASE_DIR / "fraud_results.csv"
MODEL_FILE = BASE_DIR / "isolation_forest_model.joblib"

# =========================================================
# 2. BLUE AND WHITE THEME
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
    color: #17365D;
}
[data-testid="stSidebar"] {
    background-color: #EFF6FF;
    border-right: 1px solid #D6E5F5;
}
.block-container {
    padding-top: 1.5rem;
    max-width: 1500px;
}
h1, h2, h3 {
    color: #123B68 !important;
}
[data-testid="stMetric"] {
    background: #F1F7FE;
    border: 1px solid #D6E5F5;
    border-radius: 12px;
    padding: 15px;
}
.stButton > button, .stDownloadButton > button {
    background-color: #1769AA;
    color: white;
    border-radius: 8px;
    border: none;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #123B68;
    color: white;
}
.hero {
    padding: 24px;
    border-radius: 14px;
    background: linear-gradient(120deg, #123B68, #2385D1);
    color: white;
    margin-bottom: 20px;
}
.hero h1, .hero p {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>GST Invoice Fraud Detection System</h1>
    <p>Single Invoice Checking | Excel Analysis | Isolation Forest | Risk Scoring</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 3. LOGIN AUTHENTICATION
# =========================================================
if not AUTH_FILE.exists():
    st.error("users.yaml was not found in the project folder.")
    st.info("Place your existing users.yaml file beside app.py.")
    st.stop()

try:
    with open(AUTH_FILE, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=SafeLoader)

    import streamlit_authenticator as stauth

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
    authenticator.login(location="main")

except Exception as error:
    st.error("Login configuration error.")
    st.code(str(error))
    st.stop()

auth_status = st.session_state.get("authentication_status")

if auth_status is False:
    st.error("Incorrect username or password.")
    st.stop()

if auth_status is None:
    st.info("Enter your username and password to continue.")
    st.stop()


with st.sidebar:
    st.success(f"Welcome, {st.session_state.get('name', 'User')}!")

    authenticator.logout(
        location="sidebar",
        key="unique_logout_button"
    )

    st.divider()
    st.caption("GST Invoice Analysis Dashboard")


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================
def normalize_columns(df):
    """Normalize uploaded column names."""
    result = df.copy()
    result.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in result.columns
    ]
    return result


def number(value, default=0.0):
    """Convert a value to a number safely."""
    try:
        if pd.isna(value) or str(value).strip() == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_date(value):
    """Convert a date safely."""
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT


def find_column(df, *names):
    columns = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in columns:
            return columns[name.lower()]
    return None


@st.cache_resource
def load_model(model_path, modified_time):
    """Load the saved Isolation Forest model."""
    try:
        return joblib.load(model_path)
    except Exception:
        return None


model = None

if MODEL_FILE.exists():
    model = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)

# These are the feature names used by the existing feature-engineering
# design. The model must have been trained with compatible features.
FEATURE_COLUMNS = [
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
    "invoice_number_frequency",
]


# =========================================================
# 5. GST INVOICE FEATURE ENGINEERING
# =========================================================
def make_features(row):
    quantity = number(row.get("quantity", 1), 1)
    unit_price = number(row.get("unit_price", 0))
    discount_percent = number(row.get("discount_percent", 0))
    tax_percent = number(
        row.get("tax_percent", row.get("gst_rate", 0))
    )

    calculated_subtotal = quantity * unit_price

    subtotal = number(
        row.get("subtotal", calculated_subtotal),
        calculated_subtotal,
    )

    calculated_discount = (
        calculated_subtotal * discount_percent / 100
    )

    discount_amount = number(
        row.get("discount_amount", calculated_discount),
        calculated_discount,
    )

    taxable_value = max(
        calculated_subtotal - discount_amount, 0
    )

    calculated_tax = taxable_value * tax_percent / 100

    tax_amount = number(
        row.get("tax_amount", calculated_tax),
        calculated_tax,
    )

    shipping_cost = number(row.get("shipping_cost", 0))

    calculated_total = (
        subtotal - discount_amount + tax_amount + shipping_cost
    )

    total_amount = number(
        row.get(
            "total_amount",
            row.get("invoice_amount", calculated_total),
        ),
        calculated_total,
    )

    invoice_date = parse_date(row.get("invoice_date"))
    due_date = parse_date(row.get("due_date"))
    payment_date = parse_date(row.get("payment_date"))

    payment_terms_days = 0
    payment_delay_days = 0

    if pd.notna(invoice_date) and pd.notna(due_date):
        payment_terms_days = (due_date - invoice_date).days

    if pd.notna(payment_date) and pd.notna(due_date):
        payment_delay_days = (payment_date - due_date).days

    subtotal_difference = subtotal - calculated_subtotal
    discount_difference = discount_amount - calculated_discount
    tax_difference = tax_amount - calculated_tax
    total_difference = total_amount - calculated_total

    total_error_ratio = (
        abs(total_difference) / max(abs(calculated_total), 1)
    )

    return {
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_percent": discount_percent,
        "tax_percent": tax_percent,
        "subtotal": subtotal,
        "discount_amount": discount_amount,
        "tax_amount": tax_amount,
        "shipping_cost": shipping_cost,
        "total_amount": total_amount,
        "payment_terms_days": payment_terms_days,
        "payment_delay_days": payment_delay_days,
        "subtotal_difference": subtotal_difference,
        "discount_difference": discount_difference,
        "tax_difference": tax_difference,
        "total_difference": total_difference,
        "total_error_ratio": total_error_ratio,
        "high_discount_flag": int(discount_percent > 30),
        "high_tax_flag": int(tax_percent > 28),
        "zero_tax_flag": int(tax_percent == 0 and tax_amount != 0),
        "negative_amount_flag": int(
            any(
                x < 0
                for x in [
                    quantity,
                    unit_price,
                    subtotal,
                    discount_amount,
                    tax_amount,
                    total_amount,
                ]
            )
        ),
        "payment_after_due_flag": int(payment_delay_days > 0),
        # Historical frequency values are not known for a new invoice.
        "customer_frequency": number(
            row.get("customer_frequency", 1), 1
        ),
        "vendor_frequency": number(
            row.get("vendor_frequency", 1), 1
        ),
        "product_frequency": number(
            row.get("product_frequency", 1), 1
        ),
        "invoice_number_frequency": number(
            row.get("invoice_number_frequency", 1), 1
        ),
    }


# =========================================================
# 6. INVOICE ANALYSIS AND RISK EXPLANATIONS
# =========================================================
def analyze_invoice(row, model=None):
    features = make_features(row)
    reasons = []

    # Arithmetic checks
    if abs(features["subtotal_difference"]) > 0.01:
        reasons.append(
            "Subtotal does not match quantity multiplied by unit price."
        )

    if abs(features["discount_difference"]) > 0.01:
        reasons.append(
            "Discount amount does not match the stated discount percentage."
        )

    if abs(features["tax_difference"]) > 0.01:
        reasons.append(
            "GST amount does not match the stated tax rate and calculated taxable value."
        )

    if abs(features["total_difference"]) > 0.01:
        reasons.append(
            "Invoice total does not match the calculated invoice total."
        )

    # Other review indicators
    if features["discount_percent"] > 30:
        reasons.append(
            "Discount exceeds 30%; verify the commercial reason."
        )

    if features["tax_percent"] > 28:
        reasons.append(
            "Tax rate exceeds 28%; verify the applicable GST classification and rate."
        )

    if features["zero_tax_flag"]:
        reasons.append(
            "GST rate is zero but a non-zero GST amount is recorded."
        )

    if features["negative_amount_flag"]:
        reasons.append(
            "One or more invoice amount fields contain a negative value."
        )

    if features["payment_after_due_flag"]:
        reasons.append(
            "Payment was recorded after the due date."
        )

    # Transparent fallback risk score.
    # Replace this calculation with your original WFRS formula
    # if it differs from this one.
    score = 0.0

    score += min(features["total_error_ratio"] * 100, 40)

    score += (
        20 if abs(features["tax_difference"]) > 0.01 else 0
    )
    score += (
        15 if abs(features["discount_difference"]) > 0.01 else 0
    )
    score += (
        15 if abs(features["subtotal_difference"]) > 0.01 else 0
    )
    score += (
        10 if features["discount_percent"] > 30 else 0
    )
    score += (
        10 if features["tax_percent"] > 28 else 0
    )
    score += (
        15 if features["zero_tax_flag"] else 0
    )
    score += (
        25 if features["negative_amount_flag"] else 0
    )
    score += (
        5 if features["payment_after_due_flag"] else 0
    )

    score = float(min(max(score, 0), 100))

    # Optional Isolation Forest prediction.
    anomaly = False
    model_message = "Model unavailable; only rule checks were performed."

    if model is not None:
        try:
            model_input = pd.DataFrame(
                [[features.get(c, 0) for c in FEATURE_COLUMNS]],
                columns=FEATURE_COLUMNS,
            )

            expected = getattr(model, "feature_names_in_", None)

            if expected is not None:
                model_input = model_input.reindex(
                    columns=list(expected),
                    fill_value=0,
                )

            prediction = int(model.predict(model_input)[0])
            anomaly = prediction == -1

            model_message = "Isolation Forest prediction completed."

            if anomaly:
                reasons.append(
                    "Isolation Forest identified an unusual feature pattern."
                )
                score = max(score, 50.0)

        except Exception as error:
            model_message = (
                "Isolation Forest could not score this invoice. "
                f"Rule checks were still performed. Details: {error}"
            )

    if score >= 60:
        risk = "HIGH"
    elif score >= 30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    if not reasons:
        reasons.append(
            "No configured warning was triggered. This does not guarantee GST compliance."
        )

    return {
        **features,
        "fraud_risk_score": round(score, 2),
        "risk_category": risk,
        "is_anomaly": anomaly,
        "risk_reason": "; ".join(reasons),
        "model_status": model_message,
    }


# =========================================================
# 7. READ EXCEL AND CSV FILES
# =========================================================
def read_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload an Excel or CSV file.")


def analyze_uploaded_dataframe(df, model=None):
    normalized = normalize_columns(df)
    results = []

    for index, row in normalized.iterrows():
        record = row.to_dict()
        result = analyze_invoice(record, model=model)

        invoice_id = record.get(
            "invoice_id",
            record.get("invoice_number", f"ROW-{index + 1}"),
        )

        result["invoice_id"] = invoice_id
        result["source_row"] = index + 2
        results.append(result)

    return pd.DataFrame(results)


# =========================================================
# 8. SIDEBAR MODEL STATUS
# =========================================================
with st.sidebar:
    st.subheader("System Status")

    if model is not None:
        st.success("Isolation Forest model loaded.")
    else:
        st.warning(
            "Isolation Forest unavailable. Rule-based checks remain available."
        )

    st.caption("Anomaly scores are alerts for human review.")


# =========================================================
# 9. APPLICATION TABS
# =========================================================
dashboard_tab, single_tab, upload_tab, about_tab = st.tabs(
    [
        "📊 Existing Dashboard",
        "🧾 Check Single Invoice",
        "📁 Upload Excel / CSV",
        "ℹ️ About System",
    ]
)


# =========================================================
# TAB 1: EXISTING FRAUD RESULTS DASHBOARD
# =========================================================
with dashboard_tab:
    st.subheader("Previously Analyzed Invoices")

    # Prefer the latest session batch for immediate feedback. Otherwise load
    # persistent records from Supabase, then use the local CSV as a fallback.
    session_results = st.session_state.get("batch_results")
    if isinstance(session_results, pd.DataFrame) and not session_results.empty:
        old_df = session_results.copy()
        st.success("Showing the latest invoice analysis from this session.")
    else:
        old_df = pd.DataFrame()
        try:
            old_df = load_saved_analyses(limit=1000)
            if not old_df.empty:
                st.success("Showing saved invoice analyses from Supabase (latest 1,000 records).")
        except Exception as error:
            st.warning("Could not load saved analyses from Supabase. Trying the local CSV, if available.")
            st.caption(str(error))

        if old_df.empty and DATA_FILE.exists():
            try:
                old_df = pd.read_csv(DATA_FILE, low_memory=False)
                if not old_df.empty:
                    st.info("Showing historical results from fraud_results.csv.")
            except Exception as error:
                st.error("Unable to load fraud_results.csv.")
                st.code(str(error))
                old_df = pd.DataFrame()

    if old_df.empty:
        st.info(
            "No analyzed invoices are available yet. Open the Upload Excel / CSV tab, "
            "upload your invoice file, and click Analyze Uploaded Invoices. "
            "The results will then appear here."
        )
    else:
        old_df = normalize_columns(old_df)

        score_col = find_column(
            old_df, "fraud_risk_score", "risk_score", "wfrs_score", "wfrs"
        )
        risk_col = find_column(
            old_df, "risk_category", "risk_level", "risk_class"
        )
        id_col = find_column(
            old_df, "invoice_id", "invoice_number", "invoice_no"
        )
        amount_col = find_column(
            old_df, "total_amount", "invoice_amount", "amount"
        )
        reason_col = find_column(
            old_df, "risk_reason", "suspicious_reason", "reasons", "explanation"
        )

        old_df["_score"] = (
            pd.to_numeric(old_df[score_col], errors="coerce")
            if score_col else np.nan
        )
        if risk_col:
            old_df["_risk"] = old_df[risk_col].fillna("UNKNOWN").astype(str).str.upper()
        else:
            old_df["_risk"] = pd.cut(
                old_df["_score"],
                bins=[-np.inf, 30, 60, np.inf],
                labels=["LOW", "MEDIUM", "HIGH"],
            ).astype("string").fillna("UNKNOWN")

        old_df["_id"] = (
            old_df[id_col].astype(str) if id_col else old_df.index.astype(str)
        )
        old_df["_amount"] = (
            pd.to_numeric(old_df[amount_col], errors="coerce")
            if amount_col else np.nan
        )

        available_risks = sorted(old_df["_risk"].dropna().unique().tolist())
        selected_risks = st.multiselect(
            "Filter risk category",
            available_risks,
            default=available_risks,
            key="dashboard_risks",
        )
        search = st.text_input("Search invoice number", key="dashboard_search")

        view = old_df[old_df["_risk"].isin(selected_risks)].copy()
        if search:
            view = view[view["_id"].str.contains(search, case=False, na=False)]

        total = len(view)
        high = int(view["_risk"].isin(["HIGH", "CRITICAL"]).sum())
        medium = int((view["_risk"] == "MEDIUM").sum())
        low = int((view["_risk"] == "LOW").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Invoices", f"{total:,}")
        c2.metric("High Risk", f"{high:,}")
        c3.metric("Medium Risk", f"{medium:,}")
        c4.metric("Low Risk", f"{low:,}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Risk Category Distribution")
            st.bar_chart(view["_risk"].value_counts())
        with right:
            st.markdown("#### Risk Score Distribution")
            if view["_score"].notna().any():
                st.bar_chart(view["_score"].dropna().value_counts(bins=10).sort_index())
            else:
                st.info("No risk score column was found.")

        st.markdown("#### Invoice Results")
        display_cols = ["_id", "_amount", "_score", "_risk"]
        rename_cols = {
            "_id": "Invoice ID",
            "_amount": "Invoice Amount",
            "_score": "Risk Score",
            "_risk": "Risk Category",
        }
        if reason_col:
            display_cols.append(reason_col)
            rename_cols[reason_col] = "Suspicious Reasons"

        st.dataframe(
            view[display_cols].rename(columns=rename_cols),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Filtered Results",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="filtered_fraud_results.csv",
            mime="text/csv",
            key="download_dashboard_results",
        )


# =========================================================
# TAB 2: MANUAL SINGLE-INVOICE CHECKING
# =========================================================
with single_tab:
    st.subheader("Check a Single GST Invoice")

    st.write(
        "Enter the invoice details. The system checks the arithmetic "
        "and configured risk indicators."
    )

    with st.form("single_invoice_form"):
        left, middle, right = st.columns(3)

        with left:
            invoice_id = st.text_input("Invoice Number", "INV-001")
            vendor_name = st.text_input("Vendor Name")
            vendor_gstin = st.text_input("Vendor GSTIN (optional)")
            invoice_date = st.date_input(
                "Invoice Date", value=date.today()
            )

        with middle:
            quantity = st.number_input(
                "Quantity", min_value=0.0, value=1.0
            )
            unit_price = st.number_input(
                "Unit Price (₹)", min_value=0.0, value=1000.0
            )
            discount_percent = st.number_input(
                "Discount (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
            )
            tax_percent = st.number_input(
                "GST Rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=18.0,
            )

        with right:
            subtotal = st.number_input(
                "Recorded Subtotal (₹)", min_value=0.0, value=1000.0
            )
            discount_amount = st.number_input(
                "Discount Amount (₹)", min_value=0.0, value=0.0
            )
            tax_amount = st.number_input(
                "GST Amount (₹)", min_value=0.0, value=180.0
            )
            shipping_cost = st.number_input(
                "Shipping Cost (₹)", min_value=0.0, value=0.0
            )
            total_amount = st.number_input(
                "Recorded Invoice Total (₹)",
                min_value=0.0,
                value=1180.0,
            )

        submitted = st.form_submit_button(
            "🔍 Analyze Invoice", use_container_width=True
        )

    if submitted:
        single_invoice = {
            "invoice_id": invoice_id,
            "vendor_name": vendor_name,
            "vendor_gstin": vendor_gstin,
            "invoice_date": invoice_date,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_percent": discount_percent,
            "tax_percent": tax_percent,
            "subtotal": subtotal,
            "discount_amount": discount_amount,
            "tax_amount": tax_amount,
            "shipping_cost": shipping_cost,
            "total_amount": total_amount,
        }

        result = analyze_invoice(single_invoice, model=model)
        try:
            single_df = pd.DataFrame([{
                **single_invoice,
                **result,
            }])

            save_analysis_results(single_df, str(uuid.uuid4()))
            st.session_state["batch_results"] = single_df.copy()
            st.success("Invoice analysis saved to the database.")
        except Exception as error:
            st.error("Could not save this invoice to Supabase.")
            st.error(str(error))
        st.divider()
        st.markdown("### Invoice Analysis Result")

        a, b, c = st.columns(3)

        a.metric(
            "Risk Score",
            f'{result["fraud_risk_score"]:.2f} / 100',
        )
        b.metric("Risk Category", result["risk_category"])
        c.metric(
            "Isolation Forest",
            (
                "Anomaly"
                if result["is_anomaly"]
                else ("Not Flagged" if model is not None else "Unavailable")
            ),
        )

        if result["risk_category"] == "HIGH":
            st.error("High-risk indicators found. Review this invoice.")
        elif result["risk_category"] == "MEDIUM":
            st.warning("Some indicators require manual verification.")
        else:
            st.success(
                "No major configured warnings were found. "
                "This does not guarantee GST compliance."
            )

        st.markdown("#### Risk Explanation")

        for reason in result["risk_reason"].split("; "):
            st.markdown(f"- {reason}")

        st.caption(result["model_status"])

        calculated_subtotal = quantity * unit_price
        calculated_discount = (
            calculated_subtotal * discount_percent / 100
        )
        calculated_tax = (
            max(calculated_subtotal - discount_amount, 0)
            * tax_percent / 100
        )
        calculated_total = (
            subtotal - discount_amount + tax_amount + shipping_cost
        )

        checks = pd.DataFrame([
            {
                "Validation": "Subtotal",
                "Calculated Amount": calculated_subtotal,
                "Recorded Amount": subtotal,
                "Difference": subtotal - calculated_subtotal,
            },
            {
                "Validation": "Discount",
                "Calculated Amount": calculated_discount,
                "Recorded Amount": discount_amount,
                "Difference": discount_amount - calculated_discount,
            },
            {
                "Validation": "GST Amount",
                "Calculated Amount": calculated_tax,
                "Recorded Amount": tax_amount,
                "Difference": tax_amount - calculated_tax,
            },
            {
                "Validation": "Invoice Total",
                "Calculated Amount": calculated_total,
                "Recorded Amount": total_amount,
                "Difference": total_amount - calculated_total,
            },
        ])

        st.markdown("#### Amount Validation")
        st.dataframe(
            checks.round(2),
            use_container_width=True,
            hide_index=True,
        )

        result_df = pd.DataFrame([
            {
                "invoice_id": invoice_id,
                "vendor_name": vendor_name,
                "vendor_gstin": vendor_gstin,
                **result,
            }
        ])

        st.download_button(
            "Download Invoice Analysis",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="single_invoice_analysis.csv",
            mime="text/csv",
            key="download_single",
        )


# =========================================================
# TAB 3: EXCEL / CSV BULK INVOICE CHECKING
# =========================================================
with upload_tab:
    st.subheader("Upload Excel or CSV")

    st.write(
        "Upload a file containing one invoice per row. "
        "The system analyzes each row and provides downloadable results."
    )

    template = pd.DataFrame([{
        "invoice_id": "INV-001",
        "vendor_name": "Example Vendor",
        "quantity": 1,
        "unit_price": 1000,
        "discount_percent": 0,
        "tax_percent": 18,
        "subtotal": 1000,
        "discount_amount": 0,
        "tax_amount": 180,
        "shipping_cost": 0,
        "total_amount": 1180,
    }])

    st.download_button(
        "Download Sample CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="gst_invoice_template.csv",
        mime="text/csv",
        key="download_template",
    )

    uploaded_file = st.file_uploader(
        "Choose Excel (.xlsx/.xls) or CSV (.csv)",
        type=["xlsx", "xls", "csv"],
        key="invoice_upload",
    )

    if uploaded_file is not None:
        try:
            upload_df = read_uploaded_file(uploaded_file)

            if upload_df.empty:
                st.warning("The uploaded file contains no records.")
            else:
                st.success(
                    f"Loaded {len(upload_df):,} invoices "
                    f"with {len(upload_df.columns)} columns."
                )

                with st.expander("Preview Uploaded Data"):
                    st.dataframe(
                        upload_df.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                if st.button(
                    "🔍 Analyze Uploaded Invoices",
                    type="primary",
                    key="analyze_uploaded",
                ):
                    with st.spinner("Analyzing uploaded invoices..."):
                        batch_results = analyze_uploaded_dataframe(
                            upload_df, model=model
                        )

                    st.session_state["batch_results"] = batch_results
                    batch_id = str(uuid.uuid4())

                    try:
                        save_analysis_results(batch_results, batch_id)
                        st.session_state["last_saved_batch_id"] = batch_id
                        st.success("Analysis results saved to the database.")
                    except Exception as error:
                        st.error("Could not save results to Supabase.")
                        st.error(str(error))
                    # Refresh the app so the Existing Dashboard tab can show these results.
                    st.rerun()

        except Exception as error:
            st.error("Could not read or analyze the uploaded file.")
            st.code(str(error))
            st.info(
                "For Excel files, install openpyxl using: "
                "python -m pip install openpyxl"
            )

    batch_results = st.session_state.get("batch_results")

    if isinstance(batch_results, pd.DataFrame) and not batch_results.empty:
        st.divider()
        st.markdown("### Batch Analysis Results")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Invoices Analyzed", f"{len(batch_results):,}")
        c2.metric(
            "High Risk",
            f'{(batch_results["risk_category"] == "HIGH").sum():,}',
        )
        c3.metric(
            "Medium Risk",
            f'{(batch_results["risk_category"] == "MEDIUM").sum():,}',
        )
        c4.metric(
            "Low Risk",
            f'{(batch_results["risk_category"] == "LOW").sum():,}',
        )

        risk_filter = st.multiselect(
            "Filter Results",
            ["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM", "HIGH"],
            key="batch_risk_filter",
        )

        batch_view = batch_results[
            batch_results["risk_category"].isin(risk_filter)
        ]

        st.dataframe(
            batch_view,
            use_container_width=True,
            hide_index=True,
            height=450,
        )

        st.download_button(
            "Download All Analyzed Invoices",
            data=batch_results.to_csv(index=False).encode("utf-8"),
            file_name="gst_invoice_analysis_results.csv",
            mime="text/csv",
            key="download_batch",
        )

        st.caption(
            "The download contains all analyzed invoices, "
            "including those hidden by the current filter."
        )


# =========================================================
# TAB 4: ABOUT AND LIMITATIONS
# =========================================================
with about_tab:
    st.subheader("About the GST Invoice Fraud Detection System")

    st.markdown("""
    **The system provides:**

    - Single invoice entry and analysis.
    - Excel and CSV batch invoice analysis.
    - Invoice subtotal, discount, GST amount and total checks.
    - Configured indicators for high discounts, unusual tax rates,
      negative amounts and payment delays.
    - Optional Isolation Forest anomaly detection.
    - A risk score, risk category and explanation for review.
    - Downloadable analysis results.
    """)

    st.warning("""
    **Important limitation:** This application does not directly connect
    to GSTN. It does not independently verify GSTIN status, e-invoice IRNs,
    GST returns, or the legally applicable GST rate for every item.
    Those checks require appropriate official data and integrations.
    """)

    st.info("""
    The WFRS calculation included here is a transparent fallback formula.
    It may differ from the exact formula in your original
    `src/03_fraud_model.py`. For consistent model results, use the same
    feature engineering, feature order and WFRS logic used during training.
    """)

    st.caption(
        "An anomaly or risk score is an alert for human review, "
        "not proof of fraud."
    )


# =========================================================
# 10. FOOTER
# =========================================================
st.divider()

st.caption(
    "GST Invoice Fraud Detection System | "
    "Review flagged invoices manually before taking action."
)