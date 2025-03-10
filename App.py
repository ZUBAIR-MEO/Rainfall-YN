import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Set the title of the app
st.title("Weather Data Visualization")

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
