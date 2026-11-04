
import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom styling
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
            radial-gradient(circle at 15% 0%, rgba(53, 91, 220, 0.16), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(0, 196, 167, 0.08), transparent 24%),
            #080d18;
        color: #edf2ff;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 24, 0.75);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1628 0%, #0a1020 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.13);
    }

    [data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    .brand {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.7px;
        color: #f8fafc;
    }

    .brand span {
        color: #6d8cff;
    }

    .sidebar-caption {
        color: #8793ad;
        font-size: 0.78rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    .sidebar-status {
        background: rgba(16, 185, 129, 0.10);
        border: 1px solid rgba(16, 185, 129, 0.22);
        color: #6ee7b7;
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-top: 1.2rem;
    }

    .hero {
        padding: 1.5rem 0 1.2rem 0;
    }

    .eyebrow {
        color: #7c9aff;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 2.45rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -1.5px;
        margin: 0;
    }

    .hero-title span {
        background: linear-gradient(90deg, #7c9aff, #45dfc2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #98a5bf;
        font-size: 0.96rem;
        margin-top: 0.8rem;
        max-width: 760px;
        line-height: 1.7;
    }

    .top-chip {
        display: inline-block;
        color: #a8baff;
        background: rgba(93, 124, 255, 0.11);
        border: 1px solid rgba(124, 154, 255, 0.24);
        border-radius: 999px;
        padding: 0.38rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }

    .section-heading {
        color: #f1f5ff;
        font-size: 1.08rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
    }

    .section-description {
        color: #8592ad;
        font-size: 0.82rem;
        margin-bottom: 1.1rem;
    }

    .panel {
        background: linear-gradient(145deg, rgba(18, 28, 48, 0.96), rgba(12, 20, 36, 0.96));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
    }

    .metric-card {
        background: linear-gradient(145deg, #131f35, #0e1729);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        min-height: 115px;
    }

    .metric-label {
        color: #8d9bb7;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.65rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .metric-note {
        color: #71809e;
        font-size: 0.73rem;
        margin-top: 0.35rem;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(22, 34, 58, 0.98), rgba(13, 22, 40, 0.98));
        border: 1px solid rgba(124, 154, 255, 0.25);
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.2);
    }

    .result-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .result-subtitle {
        color: #8d9bb7;
        font-size: 0.82rem;
        margin-bottom: 1.2rem;
    }

    .result-status {
        border-radius: 12px;
        padding: 1rem 1.1rem;
        font-weight: 700;
        font-size: 1rem;
        margin: 0.8rem 0 1.2rem 0;
    }

    .status-risk {
        color: #ffb4b4;
        background: rgba(239, 68, 68, 0.11);
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    .status-safe {
        color: #75e6bf;
        background: rgba(16, 185, 129, 0.10);
        border: 1px solid rgba(16, 185, 129, 0.22);
    }

    .probability-label {
        color: #9aa8c3;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }

    .probability-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.7px;
    }

    .footer-note {
        color: #687793;
        font-size: 0.75rem;
        line-height: 1.6;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
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
        box-shadow: 0 6px 22px rgba(82, 111, 255, 0.22);
        transition: all 0.2s ease;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(90deg, #6680ff, #7890ff);
        transform: translateY(-1px);
        box-shadow: 0 8px 26px rgba(82, 111, 255, 0.32);
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

    .stNumberInput button {
        color: #a8baff !important;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.12);
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.8rem;
        }
        .metric-value {
            font-size: 1.35rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">◈ Churn<span>IQ</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-caption">
            Customer intelligence platform<br>
            Powered by machine learning
        </div>
        <div class="sidebar-status">● Prediction API configured</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**WORKSPACE**")
    st.caption("Customer Risk Analysis")
    st.caption("Prediction Dashboard")

    st.markdown("---")
    st.markdown("**MODEL INFORMATION**")
    st.caption("Algorithm: Selected ML pipeline")
    st.caption("Dataset: Telco Customer Churn")
    st.caption("Task: Binary classification")

    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-caption">
            ChurnIQ helps estimate customer churn risk
            from customer profile and service information.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Hero section
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="top-chip">✦ AI-POWERED CUSTOMER ANALYTICS</div>
        <div class="eyebrow">Customer intelligence / Prediction</div>
        <h1 class="hero-title">Understand risk.<br><span>Retain customers.</span></h1>
        <div class="hero-subtitle">
            Analyze customer profiles and identify potential churn risk
            using your trained machine-learning pipeline.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")


# --------------------------------------------------
# Customer input form
# --------------------------------------------------

st.markdown(
    """
    <div class="section-heading">Customer Profile</div>
    <div class="section-description">
        Enter the customer's service, contract, and billing information.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("customer_form"):

    # Customer details
    st.markdown(
        '<div class="panel"><div class="section-heading">01 — Customer Details</div></div>',
        unsafe_allow_html=True,
    )

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
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
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

    # Service options
    st.markdown(
        """
        <div class="section-heading">02 — Service Configuration</div>
        <div class="section-description">
            Select the customer's subscribed services.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Billing
    st.markdown(
        """
        <div class="section-heading">03 — Billing Information</div>
        <div class="section-description">
            Provide monthly and cumulative customer charges.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown("")
    submitted = st.form_submit_button("✦  Analyze Customer Churn Risk")


# --------------------------------------------------
# Prediction request
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
            churn_probability = result["churn_probability"]
            no_churn_probability = result["no_churn_probability"]

            st.markdown("---")
            st.markdown(
                """
                <div class="section-heading">Prediction Analysis</div>
                <div class="section-description">
                    Model-generated estimate based on the submitted profile.
                </div>
                """,
                unsafe_allow_html=True,
            )

            status_class = (
                "status-risk" if churn_label == "Yes" else "status-safe"
            )

            status_text = (
                "⚠  Elevated churn risk detected"
                if churn_label == "Yes"
                else "✓  Customer predicted to stay"
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="eyebrow">MODEL OUTPUT</div>
                    <div class="result-title">Customer Risk Assessment</div>
                    <div class="result-subtitle">
                        Prediction generated successfully
                    </div>
                    <div class="result-status {status_class}">
                        {status_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="probability-label">CHURN PROBABILITY</div>
                        <div class="probability-value">
                            {churn_probability * 100:.2f}%
                        </div>
                        <div class="metric-note">
                            Estimated likelihood of churn
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with metric2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="probability-label">RETENTION PROBABILITY</div>
                        <div class="probability-value">
                            {no_churn_probability * 100:.2f}%
                        </div>
                        <div class="metric-note">
                            Estimated likelihood of staying
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("")

            st.markdown("**Probability Distribution**")
            st.progress(float(churn_probability))
            st.caption(
                f"Churn: {churn_probability * 100:.2f}%  ·  "
                f"Retention: {no_churn_probability * 100:.2f}%"
            )

            if churn_label == "Yes":
                st.warning(
                    "Consider reviewing this customer's contract, service "
                    "experience, and support needs. This is a model estimate, "
                    "not a guaranteed outcome."
                )
            else:
                st.success(
                    "The model currently estimates a lower churn likelihood "
                    "for this customer. Continue monitoring customer experience."
                )

            st.caption(
                "Predictions are estimates based on historical data and should "
                "not be treated as guaranteed customer behavior."
            )

        else:
            st.error(
                f"API returned status {response.status_code}: "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the FastAPI server. "
            "Ensure the API is running at http://127.0.0.1:8000."
        )

    except requests.exceptions.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown(
    """
    <div class="footer-note">
        ChurnIQ · Customer Churn Prediction<br>
        Machine-learning decision support · Not a guarantee of future behavior
    </div>
    """,
    unsafe_allow_html=True,
)
