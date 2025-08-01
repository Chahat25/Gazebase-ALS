import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data_dir = r"C:\Users\Chahat\eye_communicator\data\cnn"
model_dir =r"C:\Users\Chahat\eye_communicator\models"
os.makedirs(model_dir, exist_ok=True)


X = np.load(os.path.join(data_dir, "X.npy"))
y = np.load(os.path.join(data_dir, "y.npy"))
print("X shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Conv1D(32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2)
])

model.compile(optimizer='adam', loss='mse')
model.summary()


model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print("Mean Squared Error (x, y):", mse)


model.save(os.path.join(model_dir, "gaze_cnn.h5"))
print("Model saved:", os.path.join(model_dir, "gaze_cnn.h5"))