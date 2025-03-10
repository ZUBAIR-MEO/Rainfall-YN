import joblib
import pandas as pd
import streamlit as st
import os

# Load model
model_path = 'rainfall_model.pkl'

# Check if model file exists before loading
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.write("Model loaded successfully.")
else:
    st.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

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
missing_columns = set(expected_columns) - set(new_data.columns)
if missing_columns:
    st.error(f"Missing columns in the new data: {missing_columns}")
    raise ValueError(f"Missing columns in the new data: {missing_columns}")
else:
    new_data = new_data[expected_columns]  # Reorder the columns to match the training data order

# Ensure the number of features in the new data matches what the model expects
if new_data.shape[1] != len(expected_columns):
    st.error(f"Expected {len(expected_columns)} features, but got {new_data.shape[1]} features in the new data.")
    raise ValueError(f"Expected {len(expected_columns)} features, but got {new_data.shape[1]} features in the new data.")

# Make prediction
prediction = model.predict(new_data)

# Display prediction
st.write(f"Predicted rainfall: {prediction[0]}")
