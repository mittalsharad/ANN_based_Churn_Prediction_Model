import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Loading Trained Model
model = tf.keras.models.load_model('model.h5')

# Loading the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('one_hot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f) 


## StreamLit App
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn based on their details.")
st.write("Please enter the following details:")

### User Input
geography = st.selectbox("Geography",one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_) 
age = st.slider("Age", 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

## Create DataFrame from User Input
input_data = pd.DataFrame(
    {
        'CreditScore': [credit_score],
        'Gender' : [label_encoder_gender.transform([gender])[0]],
        'Age' : [age],
        'Tenure' : [tenure],
        'Balance' : [balance],
        'NumOfProducts' : [num_of_products],
        'HasCrCard' : [has_cr_card],
        'IsActiveMember' : [is_active_member],
        'EstimatedSalary' : [estimated_salary],
    }
)

# One Hot Encoding for Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Final Input Data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Predict Churn
prediction = model.predict(input_data)
prediction_probability = prediction[0][0]

if prediction_probability >= 0.5:
    st.write(f"Customer is likely to churn with a probability of {prediction_probability:.2f}")
else:   
    st.write(f"Customer is unlikely to churn with a probability of {prediction_probability:.2f}")
st.write("Thank you for using the Customer Churn Prediction app!")
