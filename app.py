
from pathlib import Path
from datetime import date
import json
import uuid
import os
import tempfile

import bcrypt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml

try:
    from supabase import create_client
except ImportError:
    create_client = None


# ============================================================
# 1. CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="GST Invoice Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
AUTH_FILE = BASE_DIR / "users.yaml"
MODEL_FILE = BASE_DIR / "isolation_forest_model.joblib"
RESULTS_FILE = BASE_DIR / "fraud_results.csv"

FEATURE_COLUMNS = [
    "quantity", "unit_price", "discount_percent", "tax_percent",
    "subtotal", "discount_amount", "tax_amount", "shipping_cost",
    "total_amount", "payment_terms_days", "payment_delay_days",
    "subtotal_difference", "discount_difference", "tax_difference",
    "total_difference", "total_error_ratio", "high_discount_flag",
    "high_tax_flag", "zero_tax_flag", "negative_amount_flag",
    "payment_after_due_flag", "customer_frequency", "vendor_frequency",
    "product_frequency", "invoice_number_frequency",
]


# ============================================================
# 2. BLUE AND WHITE DESIGN
# ============================================================
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
    background-color: #F1F7FE;
    border: 1px solid #D6E5F5;
    border-radius: 12px;
    padding: 14px;
}
.stButton > button,
.stDownloadButton > button {
    background-color: #1769AA;
    color: white;
    border-radius: 8px;
    border: none;
}
.hero {
    padding: 24px;
    border-radius: 14px;
    background: linear-gradient(120deg, #123B68, #2385D1);
    color: white;
    margin-bottom: 22px;
}
.hero h1, .hero p {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>GST Invoice Fraud Detection System</h1>
    <p>Invoice Validation | Isolation Forest | Risk Scoring</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 3. REGISTRATION AND LOGIN
# ============================================================
def default_auth_config():
    return {
        "credentials": {"usernames": {}}
    }


def read_auth_config():
    """Read the existing YAML user database."""
    if not AUTH_FILE.exists():
        return default_auth_config()

    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        config.setdefault("credentials", {})
        config["credentials"].setdefault("usernames", {})

        return config

    except (OSError, yaml.YAMLError):
        st.error("Could not read users.yaml. Check its YAML format.")
        st.stop()


def write_auth_config(config):
    """Write user configuration safely using a temporary file."""
    AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".yaml",
            prefix="users_",
            dir=AUTH_FILE.parent,
            delete=False,
        ) as temp_file:
            yaml.safe_dump(
                config,
                temp_file,
                sort_keys=False,
                allow_unicode=True,
            )
            temp_path = temp_file.name

        os.replace(temp_path, AUTH_FILE)

    except Exception:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def normalize_username(username):
    return username.strip().lower()


def hash_password(password):
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(),
    ).decode("utf-8")


def verify_password(password, password_hash):
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except (ValueError, TypeError, AttributeError):
        return False


def register_user(name, username, email, password):
    config = read_auth_config()
    users = config["credentials"]["usernames"]

    username = normalize_username(username)
    email = email.strip().lower()
    name = name.strip()

    if not name or not username or not email or not password:
        return False, "Please fill in every field."

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if not all(char.isalnum() or char in "._-" for char in username):
        return False, (
            "Username can contain letters, numbers, dots, "
            "underscores and hyphens only."
        )

    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Enter a valid email address."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if not any(char.isalpha() for char in password):
        return False, "Password must contain at least one letter."

    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one number."

    if username in users:
        return False, "This username already exists. Please log in."

    # Avoid allowing the same email to be registered under
    # multiple usernames in this local user database.
    for existing_user in users.values():
        if str(existing_user.get("email", "")).lower() == email:
            return False, "This email is already registered. Please log in."

    users[username] = {
        "name": name,
        "email": email,
        "password": hash_password(password),
    }

    try:
        write_auth_config(config)
    except Exception as error:
        return False, f"Could not save your account: {error}"

    return True, "Registration successful. You can now log in."


def login_user(username, password):
    config = read_auth_config()
    users = config["credentials"]["usernames"]
    username = normalize_username(username)

    if username not in users:
        return False, None, (
            "This username was not found. Please register a new account."
        )

    user = users[username]
    stored_hash = user.get("password", "")

    if not verify_password(password, stored_hash):
        return False, None, "Incorrect password. Please try again."

    return True, {
        "username": username,
        "name": user.get("name", username),
        "email": user.get("email", ""),
    }, "Login successful."


# Initialize session authentication state.
if "auth_user" not in st.session_state:
    st.session_state["auth_user"] = None

if "auth_message" not in st.session_state:
    st.session_state["auth_message"] = None

if "auth_message_type" not in st.session_state:
    st.session_state["auth_message_type"] = "info"


# Show registration/login page until authenticated.
if not st.session_state["auth_user"]:

    if st.session_state["auth_message"]:
        message = st.session_state["auth_message"]
        message_type = st.session_state["auth_message_type"]

        if message_type == "success":
            st.success(message)
        elif message_type == "error":
            st.error(message)
        else:
            st.info(message)

        st.session_state["auth_message"] = None

    st.subheader("Welcome!")
    st.write("Register a new account or log in to continue.")

    login_tab, register_tab = st.tabs([
        "🔐 Login",
        "📝 Register",
    ])

    with login_tab:
        st.markdown("### Login to your account")

        with st.form("login_form"):
            login_username = st.text_input(
                "Username",
                key="login_username",
            )

            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )

            login_submit = st.form_submit_button(
                "Login",
                use_container_width=True,
            )

        if login_submit:
            if not login_username.strip() or not login_password:
                st.error("Enter your username and password.")

            else:
                success, user, message = login_user(
                    login_username,
                    login_password,
                )

                if success:
                    st.session_state["auth_user"] = user
                    st.session_state["auth_message"] = (
                        f"Welcome, {user['name']}!"
                    )
                    st.session_state["auth_message_type"] = "success"
                    st.rerun()

                else:
                    st.error(message)

                    if "not found" in message.lower():
                        st.info(
                            "Open the Register tab to create your account."
                        )

    with register_tab:
        st.markdown("### Create a new account")

        with st.form("register_form"):
            register_name = st.text_input(
                "Full Name",
                key="register_name",
            )

            register_username = st.text_input(
                "Choose a Username",
                key="register_username",
            )

            register_email = st.text_input(
                "Email Address",
                key="register_email",
            )

            register_password = st.text_input(
                "Create Password",
                type="password",
                key="register_password",
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="confirm_password",
            )

            register_submit = st.form_submit_button(
                "Create Account",
                use_container_width=True,
            )

        if register_submit:
            if register_password != confirm_password:
                st.error("The passwords do not match.")

            else:
                success, message = register_user(
                    register_name,
                    register_username,
                    register_email,
                    register_password,
                )

                if success:
                    st.session_state["auth_message"] = message
                    st.session_state["auth_message_type"] = "success"
                    st.rerun()
                else:
                    st.error(message)

    st.divider()
    st.caption(
        "Your password is stored as a bcrypt hash, not as plain text. "
        "Keep users.yaml private."
    )

    st.stop()


# Current authenticated user.
USERNAME = st.session_state["auth_user"]["username"]
DISPLAY_NAME = st.session_state["auth_user"]["name"]

with st.sidebar:
    st.success(f"Welcome, {DISPLAY_NAME}!")

    st.caption(f"Username: {USERNAME}")

    if st.button("Log out", use_container_width=True):
        st.session_state["auth_user"] = None
        st.session_state.pop("batch_results", None)
        st.rerun()

    st.divider()
    st.caption("GST Invoice Analysis Dashboard")


# ============================================================
# 4. SUPABASE CONNECTION AND HISTORY
# ============================================================
@st.cache_resource
def get_supabase_client():
    if create_client is None:
        raise RuntimeError(
            "Supabase package missing. Install it with "
            "python -m pip install supabase"
        )

    if (
        "SUPABASE_URL" not in st.secrets
        or "SUPABASE_SECRET_KEY" not in st.secrets
    ):
        raise RuntimeError(
            "Add SUPABASE_URL and SUPABASE_SECRET_KEY "
            "to .streamlit/secrets.toml."
        )

    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SECRET_KEY"],
    )


def save_analysis_results(results_df, batch_id):
    client = get_supabase_client()

    records = json.loads(
        results_df.copy().to_json(
            orient="records",
            date_format="iso",
        )
    )

    rows = []

    for record in records:
        invoice_id = (
            record.get("invoice_id")
            or record.get("invoice_number")
            or record.get("source_row")
            or "UNKNOWN"
        )

        rows.append({
            "username": USERNAME,
            "batch_id": batch_id,
            "invoice_id": str(invoice_id),
            "invoice_data": record,
            "risk_score": record.get("fraud_risk_score"),
            "risk_level": record.get("risk_category"),
            "is_anomaly": bool(record.get("is_anomaly", False)),
            "risk_reason": str(record.get("risk_reason", "")),
        })

    for start in range(0, len(rows), 500):
        client.table("invoice_analyses").insert(
            rows[start:start + 500]
        ).execute()


def load_saved_analyses(limit=1000):
    client = get_supabase_client()

    response = (
        client.table("invoice_analyses")
        .select(
            "invoice_id, invoice_data, risk_score, risk_level, "
            "is_anomaly, risk_reason, analyzed_at, batch_id"
        )
        .eq("username", USERNAME)
        .order("analyzed_at", desc=True)
        .limit(limit)
        .execute()
    )

    records = []

    for item in response.data or []:
        payload = item.get("invoice_data") or {}

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}

        if not isinstance(payload, dict):
            payload = {}

        record = dict(payload)
        record.setdefault("invoice_id", item.get("invoice_id"))
        record.setdefault("fraud_risk_score", item.get("risk_score"))
        record.setdefault("risk_category", item.get("risk_level"))
        record.setdefault("is_anomaly", item.get("is_anomaly", False))
        record.setdefault("risk_reason", item.get("risk_reason", ""))
        record.setdefault("analyzed_at", item.get("analyzed_at"))
        record.setdefault("batch_id", item.get("batch_id"))
        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 5. LOAD ISOLATION FOREST MODEL
# ============================================================
@st.cache_resource
def load_model(model_path, modified_time):
    try:
        return joblib.load(model_path)
    except Exception:
        return None


model = None

if MODEL_FILE.exists():
    model = load_model(
        str(MODEL_FILE),
        MODEL_FILE.stat().st_mtime,
    )

with st.sidebar:
    st.markdown("### System Status")

    if model is not None:
        st.success("Isolation Forest model loaded.")
    else:
        st.warning("Model unavailable. Rule-based checks remain available.")


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================
def normalize_columns(df):
    result = df.copy()
    result.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in result.columns
    ]
    return result


def number(value, default=0.0):
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return float(default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def parse_date(value):
    return pd.to_datetime(value, errors="coerce")


def get_value(row, names, default=None):
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                if not pd.isna(value) and str(value).strip() != "":
                    return value
            except (TypeError, ValueError):
                return value
    return default


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================
def make_features(row):
    quantity = number(row.get("quantity", 1), 1)
    unit_price = number(row.get("unit_price", 0))
    discount_percent = number(row.get("discount_percent", 0))
    tax_percent = number(get_value(row, ["tax_percent", "gst_rate"], 0))

    calculated_subtotal = quantity * unit_price
    subtotal = number(
        row.get("subtotal", calculated_subtotal),
        calculated_subtotal,
    )

    calculated_discount = calculated_subtotal * discount_percent / 100
    discount_amount = number(
        row.get("discount_amount", calculated_discount),
        calculated_discount,
    )

    calculated_tax = (subtotal - discount_amount) * tax_percent / 100
    tax_amount = number(row.get("tax_amount", calculated_tax), calculated_tax)

    shipping_cost = number(row.get("shipping_cost", 0))

    calculated_total = (
        subtotal - discount_amount + tax_amount + shipping_cost
    )

    total_amount = number(
        get_value(row, ["total_amount", "invoice_amount"], calculated_total),
        calculated_total,
    )

    invoice_date = parse_date(row.get("invoice_date"))
    due_date = parse_date(row.get("due_date"))
    payment_date = parse_date(row.get("payment_date"))

    payment_terms_days = (
        (due_date - invoice_date).days
        if pd.notna(invoice_date) and pd.notna(due_date)
        else 0
    )

    payment_delay_days = (
        (payment_date - due_date).days
        if pd.notna(payment_date) and pd.notna(due_date)
        else 0
    )

    subtotal_difference = subtotal - calculated_subtotal
    discount_difference = discount_amount - calculated_discount
    tax_difference = tax_amount - calculated_tax
    total_difference = total_amount - calculated_total

    total_error_ratio = (
        abs(total_difference) / max(abs(calculated_total), 1.0)
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
        "zero_tax_flag": int(tax_percent == 0 and abs(tax_amount) > 0.01),
        "negative_amount_flag": int(
            any(value < 0 for value in [
                quantity, unit_price, subtotal, discount_amount,
                tax_amount, total_amount,
            ])
        ),
        "payment_after_due_flag": int(payment_delay_days > 0),
        "customer_frequency": number(row.get("customer_frequency", 1), 1),
        "vendor_frequency": number(row.get("vendor_frequency", 1), 1),
        "product_frequency": number(row.get("product_frequency", 1), 1),
        "invoice_number_frequency": number(
            row.get("invoice_number_frequency", 1), 1
        ),
    }


# ============================================================
# 8. INVOICE ANALYSIS AND RISK SCORING
# ============================================================
def analyze_invoice(row, fitted_model=None):
    features = make_features(row)
    reasons = []

    if abs(features["subtotal_difference"]) > 0.01:
        reasons.append("Subtotal does not match quantity multiplied by unit price.")

    if abs(features["discount_difference"]) > 0.01:
        reasons.append("Discount amount does not match the stated discount percentage.")

    if abs(features["tax_difference"]) > 0.01:
        reasons.append("GST amount does not match the calculated GST amount.")

    if abs(features["total_difference"]) > 0.01:
        reasons.append(
            "Invoice total does not match subtotal minus discount "
            "plus GST and shipping."
        )

    if features["discount_percent"] > 30:
        reasons.append("Discount exceeds the configured 30% review threshold.")

    if features["tax_percent"] > 28:
        reasons.append("Tax rate exceeds 28%; verify the applicable GST rate.")

    if features["zero_tax_flag"]:
        reasons.append("GST rate is zero but a non-zero GST amount is recorded.")

    if features["negative_amount_flag"]:
        reasons.append("A quantity or invoice amount is negative.")

    if features["payment_after_due_flag"]:
        reasons.append("Payment date is later than the due date.")

    score = min(features["total_error_ratio"] * 100, 40)

    score += 20 if abs(features["tax_difference"]) > 0.01 else 0
    score += 15 if abs(features["discount_difference"]) > 0.01 else 0
    score += 15 if abs(features["subtotal_difference"]) > 0.01 else 0
    score += 10 if features["discount_percent"] > 30 else 0
    score += 10 if features["tax_percent"] > 28 else 0
    score += 15 if features["zero_tax_flag"] else 0
    score += 25 if features["negative_amount_flag"] else 0
    score += 5 if features["payment_after_due_flag"] else 0

    anomaly = False
    model_status = "Isolation Forest unavailable; rule checks only."

    if fitted_model is not None:
        try:
            expected_columns = list(
                getattr(fitted_model, "feature_names_in_", FEATURE_COLUMNS)
            )

            model_input = pd.DataFrame(
                [[features.get(column, 0) for column in expected_columns]],
                columns=expected_columns,
            )

            prediction = int(fitted_model.predict(model_input)[0])
            anomaly = prediction == -1
            model_status = "Isolation Forest prediction completed."

            if anomaly:
                reasons.append(
                    "Isolation Forest flagged an unusual feature pattern."
                )
                score += 20

        except Exception as error:
            model_status = (
                "Isolation Forest prediction failed; rule score used. "
                + str(error)
            )

    score = round(float(np.clip(score, 0, 100)), 2)

    if score >= 60:
        risk_category = "HIGH"
    elif score >= 30:
        risk_category = "MEDIUM"
    else:
        risk_category = "LOW"

    if not reasons:
        reasons.append(
            "No configured arithmetic or threshold warning was triggered."
        )

    return {
        **features,
        "fraud_risk_score": score,
        "risk_category": risk_category,
        "is_anomaly": bool(anomaly),
        "risk_reason": "; ".join(reasons),
        "model_status": model_status,
    }


# ============================================================
# 9. BULK ANALYSIS AND FILE READING
# ============================================================
def analyze_uploaded_dataframe(df, fitted_model=None):
    clean_df = normalize_columns(df)
    results = []

    for index, row in clean_df.iterrows():
        record = row.to_dict()
        result = analyze_invoice(record, fitted_model=fitted_model)

        result["invoice_id"] = get_value(
            record,
            ["invoice_id", "invoice_number"],
            f"ROW-{index + 1}",
        )
        result["source_row"] = index + 2

        for column in [
            "invoice_number", "invoice_date", "vendor_name", "vendor_id",
            "customer_name", "customer_id", "currency", "payment_status",
        ]:
            if column in record and column not in result:
                result[column] = record[column]

        results.append(result)

    return pd.DataFrame(results)


def read_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")

    if filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload an Excel or CSV file.")


def get_risk_counts(df):
    if df.empty or "risk_category" not in df.columns:
        return {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    categories = (
        df["risk_category"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    )

    return {
        "HIGH": int(categories.isin(["HIGH", "CRITICAL"]).sum()),
        "MEDIUM": int((categories == "MEDIUM").sum()),
        "LOW": int((categories == "LOW").sum()),
    }


def save_results_safely(results):
    try:
        save_analysis_results(results, str(uuid.uuid4()))
        return True, "Results saved to Supabase."
    except Exception as error:
        return False, str(error)


# ============================================================
# 10. TABS
# ============================================================
dashboard_tab, single_tab, upload_tab, about_tab = st.tabs([
    "📊 Dashboard",
    "🧾 Single Invoice",
    "📁 Excel / CSV Upload",
    "ℹ️ About",
])


# ============================================================
# 11. DASHBOARD
# ============================================================
with dashboard_tab:
    st.subheader("Previously Analyzed Invoices")

    session_results = st.session_state.get("batch_results")

    if isinstance(session_results, pd.DataFrame) and not session_results.empty:
        display_df = session_results.copy()
        st.caption("Showing the latest analysis from this session.")
    else:
        display_df = pd.DataFrame()

        try:
            display_df = load_saved_analyses()
            if not display_df.empty:
                st.success("Loaded saved analyses from Supabase.")
        except Exception as error:
            st.warning("Could not load Supabase history.")
            st.caption(str(error))

        if display_df.empty and RESULTS_FILE.exists():
            try:
                display_df = pd.read_csv(RESULTS_FILE, low_memory=False)
                st.info("Showing results from fraud_results.csv.")
            except Exception as error:
                st.error(f"Could not read local results: {error}")

    if display_df.empty:
        st.info("No saved results yet. Analyze an invoice or upload a file.")
    else:
        display_df = normalize_columns(display_df)

        if "fraud_risk_score" not in display_df.columns:
            for candidate in ["risk_score", "wfrs_score", "wfrs"]:
                if candidate in display_df.columns:
                    display_df["fraud_risk_score"] = display_df[candidate]
                    break

        if "risk_category" not in display_df.columns:
            for candidate in ["risk_level", "risk_class"]:
                if candidate in display_df.columns:
                    display_df["risk_category"] = display_df[candidate]
                    break

        if "risk_category" not in display_df.columns:
            if "fraud_risk_score" in display_df.columns:
                scores = pd.to_numeric(
                    display_df["fraud_risk_score"], errors="coerce"
                )
                display_df["risk_category"] = np.select(
                    [scores >= 60, scores >= 30],
                    ["HIGH", "MEDIUM"],
                    default="LOW",
                )
            else:
                display_df["risk_category"] = "UNKNOWN"

        display_df["risk_category"] = (
            display_df["risk_category"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        display_df["_score"] = pd.to_numeric(
            display_df.get(
                "fraud_risk_score",
                pd.Series(np.nan, index=display_df.index),
            ),
            errors="coerce",
        )

        id_col = next(
            (
                col for col in ["invoice_id", "invoice_number", "invoice_no"]
                if col in display_df.columns
            ),
            None,
        )

        display_df["_invoice_display"] = (
            display_df[id_col].astype(str)
            if id_col
            else display_df.index.astype(str)
        )

        selected_categories = st.multiselect(
            "Filter risk category",
            options=["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
            default=["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
            key="dashboard_category_filter",
        )

        search_text = st.text_input(
            "Search invoice ID or number",
            key="dashboard_invoice_search",
        )

        filtered_df = display_df[
            display_df["risk_category"].isin(selected_categories)
        ].copy()

        if search_text.strip():
            filtered_df = filtered_df[
                filtered_df["_invoice_display"].str.contains(
                    search_text.strip(),
                    case=False,
                    na=False,
                )
            ]

        counts = get_risk_counts(filtered_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Invoices", f"{len(filtered_df):,}")
        c2.metric("High Risk", f'{counts["HIGH"]:,}')
        c3.metric("Medium Risk", f'{counts["MEDIUM"]:,}')
        c4.metric("Low Risk", f'{counts["LOW"]:,}')

        left, right = st.columns(2)

        with left:
            st.markdown("#### Risk Category Distribution")
            if not filtered_df.empty:
                category_counts = (
                    filtered_df["risk_category"]
                    .value_counts()
                    .reindex(["HIGH", "MEDIUM", "LOW", "UNKNOWN"], fill_value=0)
                )
                st.bar_chart(category_counts[category_counts > 0])

        with right:
            st.markdown("#### Risk Score Distribution")
            valid_scores = filtered_df["_score"].dropna()

            if not valid_scores.empty:
                labels = [
                    "0-10", "11-20", "21-30", "31-40", "41-50",
                    "51-60", "61-70", "71-80", "81-90", "91-100",
                ]
                edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

                score_bins = pd.cut(
                    valid_scores.clip(0, 100),
                    bins=edges,
                    labels=labels,
                    include_lowest=True,
                    right=True,
                )

                chart_df = (
                    score_bins.value_counts(sort=False)
                    .reindex(labels, fill_value=0)
                    .rename_axis("Risk Score Range")
                    .reset_index(name="Invoice Count")
                )

                st.bar_chart(
                    chart_df,
                    x="Risk Score Range",
                    y="Invoice Count",
                )
            else:
                st.info("No numeric risk scores are available.")

        st.markdown("#### Invoice Results")

        preferred_columns = [
            "_invoice_display", "total_amount", "tax_amount", "tax_percent",
            "fraud_risk_score", "risk_category", "is_anomaly", "risk_reason",
        ]

        visible_columns = [
            col for col in preferred_columns if col in filtered_df.columns
        ]

        if visible_columns:
            st.dataframe(
                filtered_df[visible_columns],
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download Filtered Results",
            data=filtered_df.drop(
                columns=["_score", "_invoice_display"],
                errors="ignore",
            ).to_csv(index=False).encode("utf-8"),
            file_name="gst_dashboard_results.csv",
            mime="text/csv",
            key="download_dashboard_results",
        )

        with st.expander("View Risk Score Diagnostics"):
            st.write("Risk category counts:")
            st.write(filtered_df["risk_category"].value_counts())
            st.write("Risk score statistics:")
            st.write(filtered_df["_score"].describe())

            diagnostic_columns = [
                col for col in [
                    "_invoice_display", "_score", "risk_category",
                    "is_anomaly", "risk_reason",
                ]
                if col in filtered_df.columns
            ]

            st.dataframe(
                filtered_df[diagnostic_columns].head(30),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# 12. SINGLE INVOICE CHECK
# ============================================================
with single_tab:
    st.subheader("Check a Single GST Invoice")

    with st.form("single_invoice_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_id = st.text_input("Invoice Number", value="INV-001")
            vendor_name = st.text_input("Vendor Name")
            vendor_gstin = st.text_input("Vendor GSTIN (optional)")
            invoice_date = st.date_input("Invoice Date", value=date.today())

        with col2:
            quantity = st.number_input("Quantity", min_value=0.0, value=1.0)
            unit_price = st.number_input("Unit Price (₹)", min_value=0.0, value=1000.0)
            discount_percent = st.number_input(
                "Discount (%)", min_value=0.0, max_value=100.0, value=0.0
            )
            tax_percent = st.number_input(
                "GST Rate (%)", min_value=0.0, max_value=100.0, value=18.0
            )

        with col3:
            subtotal = st.number_input("Recorded Subtotal (₹)", min_value=0.0, value=1000.0)
            discount_amount = st.number_input("Discount Amount (₹)", min_value=0.0, value=0.0)
            tax_amount = st.number_input("GST Amount (₹)", min_value=0.0, value=180.0)
            shipping_cost = st.number_input("Shipping Cost (₹)", min_value=0.0, value=0.0)
            total_amount = st.number_input("Recorded Invoice Total (₹)", min_value=0.0, value=1180.0)

        submit_single = st.form_submit_button(
            "🔍 Analyze Invoice",
            use_container_width=True,
        )

    if submit_single:
        invoice = {
            "invoice_id": invoice_id,
            "invoice_number": invoice_id,
            "vendor_name": vendor_name,
            "vendor_gstin": vendor_gstin,
            "invoice_date": str(invoice_date),
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

        result = analyze_invoice(invoice, fitted_model=model)
        single_df = pd.DataFrame([{**invoice, **result}])
        st.session_state["batch_results"] = single_df.copy()

        success, message = save_results_safely(single_df)

        if success:
            st.success(message)
        else:
            st.warning("Analysis completed, but Supabase saving failed.")
            st.caption(message)

        st.markdown("### Analysis Result")

        a, b, c = st.columns(3)
        a.metric("Risk Score", f'{result["fraud_risk_score"]:.2f} / 100')
        b.metric("Risk Category", result["risk_category"])
        c.metric(
            "Isolation Forest",
            "Anomaly" if result["is_anomaly"]
            else ("Not Flagged" if model is not None else "Unavailable"),
        )

        if result["risk_category"] == "HIGH":
            st.error("High-risk indicators found. Review this invoice.")
        elif result["risk_category"] == "MEDIUM":
            st.warning("Some indicators require manual verification.")
        else:
            st.success(
                "No configured warnings were triggered. "
                "This does not guarantee the invoice is genuine."
            )

        st.markdown("#### Risk Explanation")
        for reason in result["risk_reason"].split("; "):
            st.markdown(f"- {reason}")

        st.caption(result["model_status"])

        calculated_subtotal = quantity * unit_price
        calculated_discount = calculated_subtotal * discount_percent / 100
        calculated_tax = (subtotal - discount_amount) * tax_percent / 100
        calculated_total = subtotal - discount_amount + tax_amount + shipping_cost

        validation = pd.DataFrame([
            {
                "Check": "Subtotal",
                "Calculated": calculated_subtotal,
                "Recorded": subtotal,
                "Difference": subtotal - calculated_subtotal,
            },
            {
                "Check": "Discount",
                "Calculated": calculated_discount,
                "Recorded": discount_amount,
                "Difference": discount_amount - calculated_discount,
            },
            {
                "Check": "GST",
                "Calculated": calculated_tax,
                "Recorded": tax_amount,
                "Difference": tax_amount - calculated_tax,
            },
            {
                "Check": "Invoice Total",
                "Calculated": calculated_total,
                "Recorded": total_amount,
                "Difference": total_amount - calculated_total,
            },
        ])

        st.markdown("#### Arithmetic Validation")
        st.dataframe(validation.round(2), use_container_width=True, hide_index=True)

        st.download_button(
            "Download Invoice Analysis",
            data=single_df.to_csv(index=False).encode("utf-8"),
            file_name="single_invoice_analysis.csv",
            mime="text/csv",
            key="download_single_result",
        )


# ============================================================
# 13. EXCEL / CSV BULK ANALYSIS
# ============================================================
with upload_tab:
    st.subheader("Bulk Invoice Analysis")

    sample = pd.DataFrame([{
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
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="gst_invoice_template.csv",
        mime="text/csv",
        key="download_sample_template",
    )

    uploaded_file = st.file_uploader(
        "Upload Excel or CSV",
        type=["xlsx", "xls", "csv"],
        key="invoice_file_uploader",
    )

    if uploaded_file is not None:
        try:
            input_df = read_uploaded_file(uploaded_file)

            if input_df.empty:
                st.warning("The uploaded file is empty.")
            else:
                st.success(
                    f"Loaded {len(input_df):,} invoices "
                    f"and {len(input_df.columns)} columns."
                )

                with st.expander("Preview Uploaded Data"):
                    st.dataframe(
                        input_df.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                if st.button(
                    "🔍 Analyze Uploaded Invoices",
                    type="primary",
                    key="analyze_uploaded_invoices",
                ):
                    with st.spinner("Analyzing invoices..."):
                        results = analyze_uploaded_dataframe(
                            input_df,
                            fitted_model=model,
                        )

                    st.session_state["batch_results"] = results
                    success, message = save_results_safely(results)

                    if success:
                        st.success(message)
                    else:
                        st.warning("Analysis completed, but Supabase saving failed.")
                        st.caption(message)

        except Exception as error:
            st.error("Could not read or analyze the uploaded file.")
            st.code(str(error))

    results = st.session_state.get("batch_results")

    if isinstance(results, pd.DataFrame) and not results.empty:
        st.divider()
        st.markdown("### Latest Analysis Results")

        counts = get_risk_counts(results)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Invoices Analyzed", f"{len(results):,}")
        c2.metric("High Risk", f'{counts["HIGH"]:,}')
        c3.metric("Medium Risk", f'{counts["MEDIUM"]:,}')
        c4.metric("Low Risk", f'{counts["LOW"]:,}')

        st.markdown("#### Actual Risk Category Counts")
        st.bar_chart(pd.Series(counts, name="Invoice Count"))

        st.markdown("#### Risk Score Statistics")
        score_series = pd.to_numeric(
            results["fraud_risk_score"], errors="coerce"
        )
        st.dataframe(
            score_series.describe().to_frame("Risk Score"),
            use_container_width=True,
        )

        selected_risks = st.multiselect(
            "Risk category",
            ["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM", "HIGH"],
            key="bulk_risk_filter",
        )

        filtered_results = results[
            results["risk_category"].isin(selected_risks)
        ]

        st.dataframe(
            filtered_results,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        st.download_button(
            "Download All Analyzed Invoices",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="gst_invoice_analysis_results.csv",
            mime="text/csv",
            key="download_all_invoice_results",
        )

        st.caption(
            "The download contains all results, including rows hidden "
            "by the selected risk-category filter."
        )


# ============================================================
# 14. ABOUT
# ============================================================
with about_tab:
    st.subheader("About the System")

    st.write(
        "The GST Invoice Fraud Detection System screens invoice data "
        "using arithmetic checks, configured risk indicators, and "
        "Isolation Forest anomaly detection when the trained model is available."
    )

    st.markdown("""
    **Features**

    - User registration and login.
    - Single invoice analysis.
    - Excel and CSV batch analysis.
    - Subtotal, discount, GST and total validation.
    - Risk score and explanation.
    - Isolation Forest anomaly flag.
    - Optional Supabase history.
    - Downloadable results.
    """)

    st.warning(
        "The application does not directly connect to GSTN and does not "
        "independently verify GSTIN status, e-invoice IRNs, GST returns, "
        "or the legally applicable GST rate for every item. A flagged "
        "invoice is not proof of fraud."
    )

    st.info(
        "For exact consistency with the original trained model, the "
        "feature engineering and scoring must match the training code "
        "in src/02_feature_engineering.py and src/03_fraud_model.py."
    )


# ============================================================
# 15. FOOTER
# ============================================================
st.divider()

st.caption(
    "GST Invoice Fraud Detection System | "
    "Risk scores support review and are not a legal determination."
)
