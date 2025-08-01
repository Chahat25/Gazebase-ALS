import numpy as np
import pandas as pd
import os
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths
processed_dir = r"C:\Users\Chahat\eye_communicator\data\processed"
output_model_dir = r"C:\Users\Chahat\eye_communicator\models"
os.makedirs(output_model_dir, exist_ok=True)

# Load processed CSVs
def load_processed_data(processed_dir, sequence_length=30, canvas_width=600, canvas_height=400):
    sequences = []
    labels = []
    csv_files = glob.glob(os.path.join(processed_dir, "clean_*.csv"))
    if not csv_files:
        print(f"Error: No processed CSVs in {processed_dir}")
        exit()

    print(f"Found {len(csv_files)} processed CSVs")
    for idx, csv_file in enumerate(csv_files):
        print(f"Loading {idx + 1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        try:
            df = pd.read_csv(csv_file)
            if 'x' not in df.columns or 'y' not in df.columns:
                print(f"Skipping {csv_file}: Missing x or y columns")
                continue
            # Check for valid data
            if df.empty or df['x'].nunique() <= 1 or df['y'].nunique() <= 1:
                print(f"Skipping {csv_file}: Empty or constant gaze data")
                continue
            # Normalize gaze to test_landmarks.py range
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            if x_max > x_min and y_max > y_min:
                gaze_x = 0.50 + (df['x'] - x_min) / (x_max - x_min) * (0.65 - 0.50)
                gaze_y = 0.44 + (df['y'] - y_min) / (y_max - y_min) * (0.50 - 0.44)
            else:
                print(f"Skipping {csv_file}: Invalid gaze range")
                continue
            # Map to canvas (assume GazeBase screen is 1920x1080)
            screen_x = df['x'] / 1920 * canvas_width
            screen_y = df['y'] / 1080 * canvas_height
            # Create sequences
            for i in range(len(df) - sequence_length):
                seq = [[gaze_x[i+j], gaze_y[i+j]] for j in range(sequence_length)]
                label = [screen_x[i+sequence_length-1], screen_y[i+sequence_length-1]]
                sequences.append(seq)
                labels.append(label)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    return np.array(sequences), np.array(labels)

# Load data
X, y = load_processed_data(processed_dir)
if X.size == 0 or y.size == 0:
    print("Error: No valid data loaded")
    print("Check preprocessed CSVs for valid x, y values")
    exit()

# Normalize labels to [0, 1]
y[:, 0] /= 600  # Canvas width
y[:, 1] /= 400  # Canvas height

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Define LSTM model
model = Sequential([
    LSTM(64, input_shape=(30, 2), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(2)  # Output (x, y)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    os.path.join(output_model_dir, 'lstm_gazebase.h5'),
    monitor='val_loss',
    save_best_only=True
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Mean Absolute Error')
plt.legend()
plt.savefig(os.path.join(output_model_dir, 'training_history.png'))
plt.close()

# Save final model
model.save(os.path.join(output_model_dir, 'lstm_gazebase_final.h5'))
print(f"Models saved in {output_model_dir}")
