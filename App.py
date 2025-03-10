import joblib
import pandas as pd
import streamlit as st
import os
import numpy as np

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

# Define the columns to use for prediction (5 variables)
selected_columns = ['temparature', 'cloud', 'day', 'maxtemp', 'pressure']

# Title of the Streamlit app
st.title('Rainfall Prediction App')

# User inputs for the 5 variables
st.header("Enter Weather Data:")

temparature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=22.5)
cloud = st.number_input("Cloudiness", min_value=0, max_value=8, value=1)
day = st.number_input("Day (1=Monday, ..., 7=Sunday)", min_value=1, max_value=7, value=2)
maxtemp = st.number_input("Maximum Temperature (°C)", min_value=-50.0, max_value=50.0, value=30.0)
pressure = st.number_input("Pressure (hPa)", min_value=950, max_value=1050, value=1012)

# Prepare the input data as a DataFrame with only the selected columns
input_data = pd.DataFrame({
    'temparature': [temparature],
    'cloud': [cloud],
    'day': [day],
    'maxtemp': [maxtemp],
    'pressure': [pressure]
})

# Ensure the input data has the correct column order
input_data = input_data[selected_columns]

# Check if the column names match the expected columns used during training
if list(input_data.columns) != selected_columns:
    st.error("The column names in the input data do not match the expected column names.")
    raise ValueError("Column names mismatch")

# Scale the input data using the preloaded scaler
try:
    input_data_scaled = scaler.transform(input_data)
except ValueError as e:
    st.error(f"Error during scaling: {e}")
    raise e

# Make the prediction using the model
try:
    prediction = model.predict(input_data_scaled)
    
    # Ensure the prediction is a scalar value
    if isinstance(prediction, (np.ndarray)) and prediction.size > 0:
        prediction_value = prediction[0]  # Get the scalar value from the prediction array
    else:
        st.error("Prediction result is invalid or empty.")
        prediction_value = None
    
    # Display the result
    if prediction_value is not None:
        st.header("Prediction Result:")
        st.write(f"Predicted Rainfall: {prediction_value:.2f} mm")
    
except Exception as e:
    st.error(f"Error during prediction: {e}")
    raise e
