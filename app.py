import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Logistics AI Dashboard", layout="wide")

st.title("🚛 Logistics Analytics Dashboard")
st.write("Professional AI-powered tools for supply chain operations.")

# تحميل النماذج (تم تصحيح اسم ملف الـ scaler)
kmeans_model = joblib.load('kmeans_clustering_model.pkl')
scaler = joblib.load('scaler_for_clustering.pkl') 
Efficiency_model = joblib.load('Efficiency.pkl')

tab1, tab2 = st.tabs(["Trip Segmentation (Clustering)", "Driver Efficiency Prediction"])

with tab1:
    st.header("Trip Category Predictor")
    col1, col2 = st.columns(2)
    
    with col1:
        dist = st.number_input("Distance (Miles)", min_value=0.0, value=500.0, key="dist_c")
        dur = st.number_input("Duration (Hours)", min_value=0.0, value=10.0, key="dur_c")
    with col2:
        weight = st.number_input("Weight (Lbs)", min_value=0.0, value=20000.0, key="weight_c")
        idle = st.number_input("Idle Time (Hours)", min_value=0.0, value=2.0, key="idle_c")
        
    if st.button("Predict Category"):
        data = pd.DataFrame([[dist, dur, weight, idle]], 
                            columns=['actual_distance_miles', 'actual_duration_hours', 'weight_lbs', 'idle_time_hours'])
        scaled_data = scaler.transform(data)
        prediction = kmeans_model.predict(scaled_data)[0]
        
        categories = {
            0: "Light transport trips (medium distances)",
            1: "Very long distance trips",
            2: "Heavy transport trips"
        }
        st.success(f"Result: {categories[prediction]}")

with tab2:
    st.header("Driver Efficiency Predictor")
    
    
    col1, col2 = st.columns(2)
    with col1:
        driver_id = st.number_input("Driver ID", value=1)
        month = st.number_input("Month", value=1)
        dist_r = st.number_input("Actual Distance (Miles)", value=500.0, key="dist_r")
        dur_r = st.number_input("Actual Duration (Hours)", value=10.0, key="dur_r")
    with col2:
        fuel = st.number_input("Fuel Gallons Used", value=50.0)
        weight_r = st.number_input("Weight (Lbs)", value=20000.0, key="weight_r")
        trip_id = st.number_input("Trip ID", value=100)
        avg_speed = st.number_input("Average Speed", value=50.0)
        
    if st.button("Predict Efficiency"):
        
        data_r = pd.DataFrame([[driver_id, month, dist_r, dur_r, fuel, weight_r, trip_id, avg_speed]], 
                            columns=['driver_id', 'month', 'actual_distance_miles', 'actual_duration_hours', 'fuel_gallons_used', 'weight_lbs', 'trip_id', 'avg_speed'])
        
        prediction_r = Efficiency_model.predict(data_r)
        st.metric("Predicted Efficiency (MPG)", f"{prediction_r[0]:.2f}")
