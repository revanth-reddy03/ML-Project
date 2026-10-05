from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "01_Dataset" / "cleaned_dataset.csv"
MODEL_PATH = BASE_DIR / "07_Trained_Models" / "random_forest.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATASET_PATH)


st.title("Metro Traffic Prediction App")
st.write("Predict traffic flow using the trained machine learning model.")

df = load_data()

with st.sidebar:
    st.header("Model controls")
    station_id = st.number_input("Station ID", min_value=1, max_value=50, value=10)
    line_id = st.number_input("Line ID", min_value=1, max_value=10, value=3)
    train_id = st.number_input("Train ID", min_value=1, max_value=200, value=120)
    hour = st.slider("Hour", 0, 23, 8)
    day_of_week = st.slider("Day of Week", 1, 7, 2)
    weekend_flag = st.checkbox("Weekend")
    holiday_flag = st.checkbox("Holiday")
    special_event_flag = st.checkbox("Special Event")
    peak_hour_flag = st.checkbox("Peak Hour")
    passenger_count = st.number_input("Passenger Count", min_value=0, max_value=2000, value=250)
    boardings = st.number_input("Boardings", min_value=0, max_value=2000, value=180)
    alightings = st.number_input("Alightings", min_value=0, max_value=2000, value=150)
    train_frequency = st.number_input("Train Frequency per Hour", min_value=1.0, max_value=20.0, value=8.5, step=0.1)
    dwell_time = st.number_input("Average Dwell Time (sec)", min_value=10.0, max_value=120.0, value=38.0, step=0.5)
    headway = st.number_input("Average Headway (min)", min_value=1.0, max_value=20.0, value=7.0, step=0.1)
    speed = st.number_input("Average Speed (kmph)", min_value=10.0, max_value=80.0, value=40.0, step=0.1)
    occupancy = st.number_input("Occupancy Rate (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    platform_density = st.number_input("Platform Density", min_value=0.0, max_value=3.0, value=0.8, step=0.01)
    temperature = st.number_input("Temperature (C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=40.0, value=1.0, step=0.1)
    visibility = st.number_input("Visibility (km)", min_value=0.0, max_value=20.0, value=8.5, step=0.1)
    signal_delay = st.number_input("Signal Delay (sec)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
    incident_flag = st.checkbox("Incident")
    maintenance_flag = st.checkbox("Maintenance")
    energy = st.number_input("Energy Consumption (kWh)", min_value=0.0, max_value=3000.0, value=1200.0, step=10.0)

input_df = pd.DataFrame([
    {
        "Timestamp": "2025-01-01 00:00:00",
        "Station_ID": station_id,
        "Line_ID": line_id,
        "Train_ID": train_id,
        "Hour": hour,
        "Day_of_Week": day_of_week,
        "Weekend_Flag": int(weekend_flag),
        "Holiday_Flag": int(holiday_flag),
        "Special_Event_Flag": int(special_event_flag),
        "Peak_Hour_Flag": int(peak_hour_flag),
        "Passenger_Count": passenger_count,
        "Boardings": boardings,
        "Alightings": alightings,
        "Ticket_Entries": int(passenger_count * 0.75),
        "Ticket_Exits": int(passenger_count * 0.6),
        "Train_Frequency_per_Hour": train_frequency,
        "Average_Dwell_Time_sec": dwell_time,
        "Average_Headway_min": headway,
        "Average_Speed_kmph": speed,
        "Occupancy_Rate_pct": occupancy,
        "Platform_Density": platform_density,
        "Temperature_C": temperature,
        "Humidity_pct": humidity,
        "Rainfall_mm": rainfall,
        "Visibility_km": visibility,
        "Signal_Delay_sec": signal_delay,
        "Incident_Flag": int(incident_flag),
        "Maintenance_Flag": int(maintenance_flag),
        "Energy_Consumption_kWh": energy,
    }
])

if MODEL_PATH.exists():
    model = load_model()
    prediction = model.predict(input_df.drop(columns=["Timestamp"]))
    st.metric("Predicted Traffic Flow", round(float(prediction[0]), 2))
else:
    st.warning("No trained model is available yet. Train a model first.")

st.subheader("Dataset preview")
st.dataframe(df.head())
