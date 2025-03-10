import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load pre-trained model and scaler
model = joblib.load('/kaggle/working/rainfall_model.pkl')
scaler = joblib.load('/kaggle/working/scaler.pkl')

# Title of the app
st.title('Rainfall Prediction Dashboard')

# Description of the app
st.write("""
This dashboard allows you to predict the likelihood of rainfall based on weather-related features.
Input the values for each feature, and the model will predict the rainfall probability.
""")

# Input fields for user to enter data
pressure = st.number_input('Pressure (hPa)', min_value=900.0, max_value=1050.0, value=1015.9)
temparature = st.number_input('Temperature (°C)', min_value=-10.0, max_value=50.0, value=21.3)
mintemp = st.number_input('Minimum Temperature (°C)', min_value=-10.0, max_value=50.0, value=20.7)
dewpoint = st.number_input('Dewpoint (°C)', min_value=-10.0, max_value=50.0, value=20.2)
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, value=95)
sunshine = st.number_input('Sunshine (hours)', min_value=0, max_value=24, value=0)

# Convert categorical input like "rainfall" (if applicable, here we assume the model uses numerical inputs)
# Note: You can customize the input fields based on your features.
rainfall = st.selectbox('Rainfall (yes/no)', ['yes', 'no'])

# Convert 'rainfall' from 'yes'/'no' to 1/0
rainfall = 1 if rainfall == 'yes' else 0

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'temparature': [temparature],
    'pressure': [pressure],
    'mintemp': [mintemp],
    'dewpoint': [dewpoint],
    'humidity': [humidity],
    'sunshine': [sunshine],
    'rainfall': [rainfall]  # Include the rainfall variable (can be omitted depending on your model)
})

# Ensure the columns are in the same order as during model training (if needed, add missing columns)
expected_columns = ['temparature', 'pressure', 'mintemp', 'dewpoint', 'humidity', 'sunshine', 'rainfall']
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing columns with default value

# Standardize the input data using the same scaler
input_data_scaled = scaler.transform(input_data)

# Predict the likelihood of rainfall
predicted_rainfall = model.predict(input_data_scaled)

# Display the result
st.write(f"**Predicted Rainfall: {'Yes' if predicted_rainfall[0] == 1 else 'No'}**")

# Visualization Section (optional)
st.subheader("Feature Input Data")
st.write(input_data)

