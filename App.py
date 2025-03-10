import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('/kaggle/working/rainfall_model.pkl')
scaler = joblib.load('/kaggle/working/scaler.pkl')

# Define the column names used during model training
expected_columns = ['temparature', 'cloud', 'day', 'maxtemp', 'winddirection', 'humidity', 'pressure', 'sunshine']

# Simulate user input (replace this with actual Streamlit inputs)
new_data = pd.DataFrame({
    'temparature': [30],
    'cloud': [20],
    'day': [1],
    'maxtemp': [35],
    'winddirection': [5],
    'humidity': [60],
    'pressure': [1015],
    'sunshine': [8]
})

# Ensure the input data has the same columns as expected
new_data = new_data[expected_columns]  # Reorder columns to match the training data

# Preprocess the new data (standardize using the same scaler)
new_data_scaled = scaler.transform(new_data)

# Make prediction using the trained model
prediction = model.predict(new_data_scaled)

# Output the prediction (this would be displayed in Streamlit)
st.write(f"Predicted Rainfall: {prediction[0]}")
