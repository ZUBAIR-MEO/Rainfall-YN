import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (Replace this with your actual dataset)
data = {
    'temperature': [22.1, 23.4, 19.8, 21.2, 25.1],
    'humidity': [85, 78, 90, 82, 88],
    'sunshine': [7, 8, 6, 5, 9],
    'pressure': [1012, 1013, 1011, 1014, 1010],
    'maxtemp': [25, 26, 24, 23, 27],
    'mintemp': [18, 19, 17, 16, 20],
    'dewpoint': [18.5, 19.2, 17.0, 18.0, 20.1],
    'rain': [0, 1, 0, 1, 0]  # Target variable (0 = no rain, 1 = rain)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define features and target for the model
features = ['temperature', 'humidity', 'sunshine', 'pressure', 'maxtemp', 'mintemp', 'dewpoint']
X = df[features]  # Features
y = df['rain']    # Target: rain (1 or 0)

# Standardizing the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a RandomForest model for rain prediction
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Output the accuracy of the model
print(f"Accuracy of Rain Prediction Model with StandardScaler: {accuracy * 100:.2f}%")
