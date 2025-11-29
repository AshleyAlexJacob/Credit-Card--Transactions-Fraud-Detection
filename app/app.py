import pandas as pd
import streamlit as st
from datetime import datetime
import pickle
import sys
import os

# Add project root (parent directory of 'app' and 'src') to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.model.inference import ModelInference
from utils import convert_dicitonary_to_df


# ======================= Load Model =============
model = ModelInference(model_name="rf_classifier_model.pkl")

# ======================= Lists =============

categories = ['personal_care', 'health_fitness', 'misc_pos', 'travel',
       'kids_pets', 'shopping_pos', 'food_dining', 'home',
       'entertainment', 'shopping_net', 'misc_net', 'grocery_pos',
       'gas_transport', 'grocery_net']
merchants = pickle.load(open("artifacts/maps/merchants_list.pkl", 'rb'))

states = pickle.load(open("artifacts/maps/states_list.pkl", 'rb'))

jobs = pickle.load(open("artifacts/maps/jobs_list.pkl", 'rb'))

genders = ['M', 'F']
# ======================= Streamlit App UI =============
st.title("Credit Card Transactions - Fraud Detection System")

amt= st.number_input("Amount")
# could be searchable drop down
zip =  st.number_input("Zip Code")
city_pop =  st.number_input("City Population")
date = st.date_input("Select date")
time = st.time_input("Select time")

date_time = datetime.combine(date, time)

year = date_time.year
month = date_time.month
day = date_time.day
hour = date_time.hour
day_of_week = date_time.weekday()


unix_time=datetime.timestamp(date_time)

merchant = st.selectbox("Select Merchant", merchants)

merchant_encoded = model.target_encoder.transform(pd.DataFrame({"merchant": [merchant], "is_fraud":[0]}))

category = st.selectbox("Select Category", categories)

gender = st.selectbox("Select Gender", genders)

job = st.selectbox("Select Job", jobs)

state = st.selectbox("Select State", states)

dob = st.date_input("Select DOB")

age = datetime.today().date().year - dob.year

distance_km = st.number_input("Distance in KM")
# ======================= Inference ===================
inference_dict = {
       "amt": amt,
       "zip": zip,
       "city_pop": city_pop,
       "unix_time": unix_time,
       "year": year,
       "month": month,
       "day": day,
       "hour": hour,
       "dayofweek": day_of_week,
       "merchant_encoded": merchant_encoded,
       "category_encoded": category,
       'gender_encoded': gender,
       'job_encoded': job,
       'state_encoded': state,
       'age': age,
       'distance_km': distance_km
}

inference_df = convert_dicitonary_to_df(inference_dict)

st.dataframe(inference_df)

# Predict Button
if st.button("Predict Fraud"):
    prediction = model.predict(inference_df)
    if prediction[0] == 1:
        st.error("Fraudulent transaction detected!")
    else:
        st.success("Legitimate transaction.")
