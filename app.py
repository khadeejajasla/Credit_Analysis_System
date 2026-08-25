import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_employment = pickle.load(open('le_employment.pkl', 'rb'))
le_eligible = pickle.load(open('le_eligible.pkl', 'rb'))

st.title('Credit Eligibility Prediction')

income = st.number_input('Income', min_value=0)
employment_status = st.selectbox('Employment Status', le_employment.classes_)
credit_score = st.number_input('Credit Score', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
repayment_period = st.number_input('Repayment Period', min_value=0)

if st.button('Predict'):
    employment_encoded = le_employment.transform([employment_status])[0]
    
    input_data = np.array([[income, employment_encoded, credit_score, loan_amount, repayment_period]],dtype=float)
    input_data[:, [0, 2, 3, 4]] = scaler.transform(input_data[:, [0, 2, 3, 4]])
    
    prediction = model.predict(input_data)
    result = le_eligible.inverse_transform(prediction)[0]
    
    st.write('Prediction:', result)