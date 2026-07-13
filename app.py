import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Logistics AI Dashboard", layout="wide")

st.title("🚛 Logistics Analytics Dashboard")
st.write("Professional AI-powered tools for supply chain operations.")

# تحميل الموديل والـ Scaler اللي عملنالهم تدريب وموجودين عندنا
kmeans_model = joblib.load('kmeans.pkl')
scaler = joblib.load('scaler.pkl')

st.header("Trip Category Predictor (Clustering)")

col1, col2 = st.columns(2)

with col1:
    dist = st.number_input("Distance (Miles)", min_value=0.0, value=500.0, key="dist_c")
    dur = st.number_input("Duration (Hours)", min_value=0.0, value=10.0, key="dur_c")
with col2:
    weight = st.number_input("Weight (Lbs)", min_value=0.0, value=20000.0, key="weight_c")
    idle = st.number_input("Idle Time (Hours)", min_value=0.0, value=2.0, key="idle_c")
    
if st.button("Predict Category"):
    # تجهيز الداتا زي ما الموديل متدرب عليها بالظبط
    data = pd.DataFrame([[dist, dur, weight, idle]], 
                        columns=['actual_distance_miles', 'actual_duration_hours', 'weight_lbs', 'idle_time_hours'])
    
    # عمل Scaling
    scaled_data = scaler.transform(data)
    
    # التنبؤ
    prediction = kmeans_model.predict(scaled_data)[0]
    
    categories = {
        0: "Light transport trips (medium distances)",
        1: "Very long distance trips",
        2: "Heavy transport trips"
    }
    
    st.success(f"Result: {categories[prediction]}")
