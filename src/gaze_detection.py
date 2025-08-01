import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Paths
webcam_data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/webcam/webcam_data.csv"))
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
os.makedirs(model_dir, exist_ok=True)

# Load webcam data
if not os.path.exists(webcam_data_file):
    print(f"Error: {webcam_data_file} not found")
    exit()

df = pd.read_csv(webcam_data_file)
print("Webcam data shape:", df.shape)
print("Sample rows:\n", df.head())

# Prepare data
X = df[['pupil_x', 'pupil_y']].values  # Features
y = df[['gaze_x', 'gaze_y']].values    # Targets

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print("Mean Squared Error (gaze_x, gaze_y):", mse)

# Save model (placeholder)
print("Model trained on webcam data. Ready for GUI testing.")