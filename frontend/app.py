from pathlib import Path

import requests
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(53, 91, 220, 0.13), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(0, 196, 167, 0.07), transparent 24%),
            #080d18;
        color: #edf2ff;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 24, 0.75);
    }

    .brand {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
    }

    .brand span {
        color: #7c9aff;
    }

    .hero {
        padding: 1rem 0 0.8rem 0;
    }

    .eyebrow {
        color: #7c9aff;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -1.2px;
        margin: 0;
    }

    .hero-title span {
        color: #7c9aff;
    }

    .hero-subtitle {
        color: #98a5bf;
        font-size: 0.92rem;
        margin-top: 0.65rem;
        max-width: 720px;
        line-height: 1.6;
    }

    .section-heading {
        color: #f1f5ff;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .section-description {
        color: #8592ad;
        font-size: 0.82rem;
        margin-bottom: 1rem;
    }

    .panel {
        background: linear-gradient(145deg, rgba(18, 28, 48, 0.96), rgba(12, 20, 36, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #131f35, #0e1729);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        min-height: 105px;
    }

    .metric-label {
        color: #8d9bb7;
        font-size: 0.76rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 800;
    }

    div[data-testid="stForm"] {
        background: transparent;
        border: none;
        padding: 0;
    }

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3rem;
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        background: linear-gradient(90deg, #526fff, #657fff);
    }

    div[data-testid="stWidgetLabel"] p {
        color: #b5c1d9;
        font-size: 0.8rem;
        font-weight: 600;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #111b2e;
        border-color: rgba(148, 163, 184, 0.18);
        border-radius: 8px;
    }

    .footer-note {
        color: #687793;
        font-size: 0.75rem;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        return None

    df = pd.read_csv(DATASET_PATH)

    # Normalize whitespace in column names
    df.columns = df.columns.str.strip()

    # Convert common numeric columns
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def show_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def churn_rate_by(df, column):
    grouped = (
        df.groupby(column, dropna=False)["Churn"]
        .apply(lambda values: (values == "Yes").mean() * 100)
        .sort_values(ascending=False)
        .to_frame("Churn Rate (%)")
    )
    return grouped


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="brand">◈ Churn<span>IQ</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Customer Intelligence</div>
        <h1 class="hero-title">Understand risk.<br><span>Retain customers.</span></h1>
        <div class="hero-subtitle">
            Explore customer churn patterns and estimate churn risk
            using a trained machine-learning model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# --------------------------------------------------
# Tabs
# --------------------------------------------------

overview_tab, analytics_tab, prediction_tab = st.tabs(
    ["Overview", "Analytics", "Predict Churn"]
)


# --------------------------------------------------
# Overview
# --------------------------------------------------

with overview_tab:
    st.markdown(
        '<div class="section-heading">Customer Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">Summary of the Telco Customer Churn dataset.</div>',
        unsafe_allow_html=True,
    )

    df = load_dataset()

    if df is None:
        st.warning(
            "Dataset not found. Expected location: "
            "`data/raw/dataset.csv`"
        )
    elif "Churn" not in df.columns:
        st.error("The dataset does not contain the expected `Churn` column.")
    else:
        total_customers = len(df)
        churned_customers = int((df["Churn"] == "Yes").sum())
        retained_customers = int((df["Churn"] == "No").sum())
        churn_rate = (
            churned_customers / total_customers * 100
            if total_customers else 0
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            show_metric("TOTAL CUSTOMERS", f"{total_customers:,}")
        with c2:
            show_metric("CHURNED", f"{churned_customers:,}")
        with c3:
            show_metric("RETAINED", f"{retained_customers:,}")
        with c4:
            show_metric("CHURN RATE", f"{churn_rate:.1f}%")

        st.markdown("")
        st.markdown("**Customer Churn Distribution**")

        distribution = (
            df["Churn"]
            .value_counts()
            .reindex(["No", "Yes"])
            .fillna(0)
            .astype(int)
        )
        distribution.index = ["Retained", "Churned"]

        st.bar_chart(distribution, use_container_width=True)


# --------------------------------------------------
# Analytics
# --------------------------------------------------

with analytics_tab:
    st.markdown(
        '<div class="section-heading">Churn Analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">Explore churn patterns across customer groups.</div>',
        unsafe_allow_html=True,
    )

    df = load_dataset()

    if df is None:
        st.warning(
            "Analytics dataset not found. Expected location: "
            "`data/raw/dataset.csv`"
        )
    elif "Churn" not in df.columns:
        st.error("The dataset does not contain the expected `Churn` column.")
    else:
        left, right = st.columns(2)

        # Churn by contract
        with left:
            st.markdown("**Churn Rate by Contract**")
            if "Contract" in df.columns:
                contract_rates = churn_rate_by(df, "Contract")
                st.bar_chart(contract_rates, use_container_width=True)
            else:
                st.info("Contract column not available.")

        # Churn by internet service
        with right:
            st.markdown("**Churn Rate by Internet Service**")
            if "InternetService" in df.columns:
                internet_rates = churn_rate_by(df, "InternetService")
                st.bar_chart(internet_rates, use_container_width=True)
            else:
                st.info("InternetService column not available.")

        left, right = st.columns(2)

        # Churn by tenure groups
        with left:
            st.markdown("**Churn Rate by Tenure**")
            if "tenure" in df.columns:
                tenure_df = df[["tenure", "Churn"]].dropna().copy()
                tenure_df["Tenure Group"] = pd.cut(
                    tenure_df["tenure"],
                    bins=[-1, 12, 24, 48, 72, 120],
                    labels=[
                        "0–12 months",
                        "13–24 months",
                        "25–48 months",
                        "49–72 months",
                        "73+ months",
                    ],
                )

                tenure_rates = churn_rate_by(tenure_df, "Tenure Group")
                tenure_rates = tenure_rates.reindex(
                    [
                        "0–12 months",
                        "13–24 months",
                        "25–48 months",
                        "49–72 months",
                        "73+ months",
                    ]
                ).dropna()

                st.bar_chart(tenure_rates, use_container_width=True)
            else:
                st.info("Tenure column not available.")

        # Monthly charges by churn status
        with right:
            st.markdown("**Average Monthly Charges**")
            if "MonthlyCharges" in df.columns:
                charges = (
                    df.groupby("Churn")["MonthlyCharges"]
                    .mean()
                    .reindex(["No", "Yes"])
                    .dropna()
                    .to_frame("Average Monthly Charges")
                )
                charges.index = charges.index.map(
                    {"No": "Retained", "Yes": "Churned"}
                )
                st.bar_chart(charges, use_container_width=True)
            else:
                st.info("MonthlyCharges column not available.")

        st.caption(
            "Churn rates are calculated from the historical dataset. "
            "These charts describe patterns and do not establish causation."
        )


# --------------------------------------------------
# Prediction form
# --------------------------------------------------

with prediction_tab:
    st.markdown(
        '<div class="section-heading">Customer Risk Assessment</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">Enter customer information and submit it to the prediction API.</div>',
        unsafe_allow_html=True,
    )

    with st.form("customer_form"):

        st.markdown("**01 — Customer Details**")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

        with col2:
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=100, value=12
            )
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )

        with col3:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.markdown("---")
        st.markdown("**02 — Service Configuration**")

        col4, col5, col6 = st.columns(3)

        with col4:
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["No", "Yes", "No internet service"]
            )

        with col5:
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )

        with col6:
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )

        st.markdown("---")
        st.markdown("**03 — Billing Information**")

        col7, col8 = st.columns(2)

        with col7:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                value=70.0,
                step=1.0,
            )

        with col8:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=840.0,
                step=1.0,
            )

        submitted = st.form_submit_button("Analyze Churn Risk")


    # --------------------------------------------------
    # Prediction request and result
    # --------------------------------------------------

    if submitted:
        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            with st.spinner("Analyzing customer profile..."):
                response = requests.post(
                    API_URL,
                    json=customer_data,
                    timeout=30,
                )

            if response.status_code == 200:
                result = response.json()

                churn_label = result["churn_label"]
                churn_probability = float(result["churn_probability"])
                no_churn_probability = float(result["no_churn_probability"])

                st.markdown("---")
                st.markdown("**Prediction Result**")

                if churn_label == "Yes":
                    st.error("Elevated churn risk detected")
                else:
                    st.success("Customer predicted to stay")

                metric1, metric2 = st.columns(2)

                with metric1:
                    show_metric(
                        "CHURN PROBABILITY",
                        f"{churn_probability * 100:.2f}%",
                    )

                with metric2:
                    show_metric(
                        "RETENTION PROBABILITY",
                        f"{no_churn_probability * 100:.2f}%",
                    )

                st.markdown("**Churn Probability**")
                st.progress(max(0.0, min(1.0, churn_probability)))

                st.caption(
                    "Predictions are estimates based on historical data, "
                    "not guaranteed customer behavior."
                )

            else:
                st.error(
                    f"API returned status {response.status_code}: "
                    f"{response.text}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Unable to connect to FastAPI. Ensure the API is running "
                "at http://127.0.0.1:8000."
            )

        except requests.exceptions.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")

        except (KeyError, ValueError, TypeError) as exc:
            st.error(f"Unexpected prediction response: {exc}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer-note">
        ChurnIQ · Customer Churn Prediction
    </div>
    """,
    unsafe_allow_html=True,
)