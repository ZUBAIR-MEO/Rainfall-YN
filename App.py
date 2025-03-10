import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load weather data
data = {
    'day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # example days
    'pressure': [1025.9, 1022, 1019.7, 1018.9, 1015.9, 1018.8, 1021.8, 1020.8, 1020.6, 1017.5],  # example pressures
    'maxtemp': [19.9, 21.7, 20.3, 22.3, 21.3, 24.3, 21.4, 21, 18.9, 18.5],  # example max temps
    'mintemp': [18.3, 18.9, 19.3, 20.6, 20.7, 20.9, 18.8, 18.4, 18.1, 18],  # example min temps
    'temperature': [16.8, 17.2, 18, 19.1, 20.2, 19.2, 17, 16.5, 17.1, 17.2],  # example temperatures
    'dewpoint': [13.1, 15.6, 18.4, 18.8, 19.9, 18, 15, 14.4, 14.3, 15.5],  # example dewpoint
    'humidity': [72, 81, 95, 90, 95, 84, 79, 78, 78, 85],  # example humidity
    'cloud': ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'yes'],  # example cloud presence
    'rainfall': [9.3, 0.6, 0, 1, 0, 7.7, 3.4, 7.7, 3.3, 0],  # example rainfall (mm)
    'sunshine': [1.1, 0, 0, 1, 0, 7.7, 3.9, 9.1, 0.2, 3.6],  # example sunshine (hours)
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Feature engineering for rain prediction
df['rain'] = df['rainfall'].apply(lambda x: 1 if x > 0 else 0)  # Binary rain column (1 for rain, 0 for no rain)

# Define features and target for the model
features = ['temperature', 'humidity', 'sunshine', 'pressure', 'maxtemp', 'mintemp', 'dewpoint']
X = df[features]  # Features
y = df['rain']    # Target: rain (1 or 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest model for rain prediction
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Set the title of the app
st.title("Weather Data Visualization and Rain Prediction")

# Show raw data
st.subheader("Raw Weather Data")
st.write(df)

# Plotting maximum temperature
st.subheader("Maximum Temperature Over Days")
plt.figure(figsize=(10, 6))
sns.lineplot(x='day', y='maxtemp', data=df, marker='o')
plt.title('Maximum Temperature vs. Day')
plt.xlabel('Day')
plt.ylabel('Max Temperature (°C)')
st.pyplot(plt)

# Plotting minimum temperature
st.subheader("Minimum Temperature Over Days")
plt.figure(figsize=(10, 6))
sns.lineplot(x='day', y='mintemp', data=df, marker='o', color='red')
plt.title('Minimum Temperature vs. Day')
plt.xlabel('Day')
plt.ylabel('Min Temperature (°C)')
st.pyplot(plt)

# Plotting humidity
st.subheader("Humidity Over Days")
plt.figure(figsize=(10, 6))
sns.lineplot(x='day', y='humidity', data=df, marker='o', color='green')
plt.title('Humidity vs. Day')
plt.xlabel('Day')
plt.ylabel('Humidity (%)')
st.pyplot(plt)

# Plotting rainfall
st.subheader("Rainfall Over Days")
plt.figure(figsize=(10, 6))
sns.barplot(x='day', y='rainfall', data=df, palette='Blues')
plt.title('Rainfall vs. Day')
plt.xlabel('Day')
plt.ylabel('Rainfall (mm)')
st.pyplot(plt)

# Plotting sunshine
st.subheader("Sunshine Hours Over Days")
plt.figure(figsize=(10, 6))
sns.barplot(x='day', y='sunshine', data=df, palette='YlOrBr')
plt.title('Sunshine Hours vs. Day')
plt.xlabel('Day')
plt.ylabel('Sunshine (Hours)')
st.pyplot(plt)

# Display cloud presence information
st.subheader("Cloud Presence")
cloud_data = df['cloud'].value_counts()
st.write("Cloud Presence Summary:")
st.write(cloud_data)

# Show the weather forecast for each day
st.subheader("Weather Forecast Summary")
weather_summary = df[['day', 'temperature', 'humidity', 'cloud', 'rainfall', 'sunshine']]
st.write(weather_summary)

# Show model accuracy
st.subheader("Rain Prediction Model Accuracy")
st.write(f"Accuracy of Rain Prediction Model: {accuracy * 100:.2f}%")

# Input for predicting rain on a new day
st.subheader("Predict Rain on New Day")

temperature_input = st.number_input("Enter temperature (°C):", min_value=-30, max_value=50)
humidity_input = st.number_input("Enter humidity (%):", min_value=0, max_value=100)
sunshine_input = st.number_input("Enter sunshine (hours):", min_value=0, max_value=24)
pressure_input = st.number_input("Enter pressure (hPa):", min_value=900, max_value=1100)
maxtemp_input = st.number_input("Enter maximum temperature (°C):", min_value=-30, max_value=50)
mintemp_input = st.number_input("Enter minimum temperature (°C):", min_value=-30, max_value=50)
dewpoint_input = st.number_input("Enter dewpoint (°C):", min_value=-30, max_value=50)

# Create a dataframe for the input data
input_data = pd.DataFrame([[temperature_input, humidity_input, sunshine_input, pressure_input, maxtemp_input, mintemp_input, dewpoint_input]],
                          columns=features)

# Predict rain based on the user input
prediction = model.predict(input_data)

if prediction == 1:
    st.write("Prediction: Rain is likely!")
else:
    st.write("Prediction: No rain expected.")
