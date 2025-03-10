import joblib
import pandas as pd
import streamlit as st

# Load model and scaler
model_path = 'rainfall_model.pkl'
scaler_path = 'scaler.pkl'

# Check if model and scaler files exist before loading
if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.write("Model and scaler loaded successfully.")
else:
    st.error(f"Model or scaler file not found at {model_path} or {scaler_path}")
    raise FileNotFoundError(f"Model or scaler file not found at {model_path} or {scaler_path}")

# Define expected columns (from the model's training data)
expected_columns = ['temparature', 'cloud', 'day', 'maxtemp', 'winddirection', 'pressure', 'humidity']

# New input data (ensure it has the same structure as the training data)
new_data = pd.DataFrame({
    'temparature': [22.5],
    'cloud': [1],
    'day': [2],
    'maxtemp': [30.0],
    'winddirection': [5],
    'pressure': [1012],
    'humidity': [85]
})

# Ensure the new data has the same columns as expected (both the same features and the correct order)
if set(new_data.columns) == set(expected_columns):
    new_data = new_data[expected_columns]  # Reorder the columns to match the training data order
else:
    missing_columns = set(expected_columns) - set(new_data.columns)
    st.error(f"Missing columns in the new data: {missing_columns}")
    raise ValueError(f"Missing columns in the new data: {missing_columns}")

# Preprocess the new data (standardize using the same scaler)
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)

# Display prediction
st.write(f"Predicted rainfall: {prediction[0]}")
