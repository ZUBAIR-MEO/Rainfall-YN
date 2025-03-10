import joblib
import pandas as pd
import streamlit as st
import os

# Set paths for the model and scaler
model_path = 'rainfall_model.pkl'
scaler_path = 'scaler.pkl'

# Load the model and scaler if they exist
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.write("Model and Scaler loaded successfully.")
else:
    st.error(f"Model or scaler files not found. Please check the paths: {model_path} and {scaler_path}")
    raise FileNotFoundError("Model or scaler files not found.")

# Define the expected columns for the model
expected_columns = ['temparature', 'cloud', 'day', 'maxtemp', 'winddirection', 'pressure', 'humidity']

# Title of the Streamlit app
st.title('Rainfall Prediction App')

# User inputs
st.header("Enter Weather Data:")
temparature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=22.5)
cloud = st.number_input("Cloudiness", min_value=0, max_value=8, value=1)
day = st.number_input("Day (1=Monday, ..., 7=Sunday)", min_value=1, max_value=7, value=2)
maxtemp = st.number_input("Maximum Temperature (°C)", min_value=-50.0, max_value=50.0, value=30.0)
winddirection = st.number_input("Wind Direction", min_value=0, max_value=360, value=5)
pressure = st.number_input("Pressure (hPa)", min_value=950, max_value=1050, value=1012)
humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=85)

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'temparature': [temparature],
    'cloud': [cloud],
    'day': [day],
    'maxtemp': [maxtemp],
    'winddirection': [winddirection],
    'pressure': [pressure],
    'humidity': [humidity]
})

# Ensure the input data has the correct column order and includes all the required features
input_data = input_data[expected_columns]  # Reorder columns to match the training data order

# Handle the missing columns and check if the columns match
missing_columns = set(expected_columns) - set(input_data.columns)
if missing_columns:
    st.error(f"Missing columns in the input data: {missing_columns}")
    raise ValueError(f"Missing columns in the input data: {missing_columns}")

# Scale the input data using the preloaded scaler
input_data_scaled = scaler.transform(input_data)

# Make the prediction using the model
prediction = model.predict(input_data_scaled)

# Display the result
st.header("Prediction Result:")
st.write(f"Predicted Rainfall: {prediction[0]:.2f} mm")
