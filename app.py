import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Credit Eligibility", page_icon="💳", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #1C2630; padding: 15px; border-radius: 10px; }
    [data-testid="stMetricValue"] { font-size: 22px; }
    h1, h2, h3 { color: #4A90D9; }
    div.stButton > button {
        background-color: #4A90D9;
        color: white;
        border-radius: 8px;
        height: 2.5em;
        font-size: 16px;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #3A7BC8;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_employment = pickle.load(open('le_employment.pkl', 'rb'))
le_eligible = pickle.load(open('le_eligible.pkl', 'rb'))

# Header
st.markdown("<h1 style='text-align: center;'> Credit Eligibility Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter applicant details below to check loan eligibility instantly</p>", unsafe_allow_html=True)
st.write("")

# Input card
with st.container(border=True):
    st.subheader("📋 Applicant Details")
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input('💰 Monthly Income (₹)', min_value=0, step=1000)
        credit_score = st.number_input('📊 Credit Score', min_value=0, max_value=900, step=10)
        repayment_period = st.number_input('📅 Repayment Period (months)', min_value=0, step=1)
    with col2:
        employment_status = st.selectbox('👤 Employment Status', le_employment.classes_)
        loan_amount = st.number_input('💵 Loan Amount (₹)', min_value=0, step=1000)

    st.write("")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        predict_clicked = st.button('🔍 Check Eligibility', use_container_width=True)

st.write("")

if predict_clicked:
    employment_encoded = le_employment.transform([employment_status])[0]

    input_data = np.array([[income, employment_encoded, credit_score, loan_amount, repayment_period]], dtype=float)
    input_data[:, [0, 2, 3, 4]] = scaler.transform(input_data[:, [0, 2, 3, 4]])

    prediction = model.predict(input_data)
    result = le_eligible.inverse_transform(prediction)[0]

    with st.container(border=True):
        st.subheader("Result")

        if result.lower() == 'yes':
            st.success("✅ **Eligible** — This applicant meets the criteria for loan approval")
        else:
            st.error("❌ **Not Eligible** — This applicant does not currently meet the criteria")

        st.write("")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Income", f"₹{income:,.0f}")
        c2.metric("Credit Score", credit_score)
        c3.metric("Loan Amount", f"₹{loan_amount:,.0f}")
        c4.metric("Repayment", f"{repayment_period} mo")

        st.write("")
        score_pct = min(credit_score / 900, 1.0)
        st.caption("Credit Score Strength")
        st.progress(score_pct)

st.write("")
st.divider()
st.caption("⚠️ This is a predictive model for educational purposes and not a substitute for actual bank credit assessment.")