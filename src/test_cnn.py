import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


model = load_model(
    r"C:\Users\Chahat\eye_communicator\models\gaze_cnn.h5",
    custom_objects={'mse': MeanSquaredError()}
)


csv_path = r"data\processed\clean_S_1077_S1_VD1.csv"
df = pd.read_csv(csv_path)
data = df[['x', 'y']].values
print("Loaded rows:", len(data))


data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)


sequence_length = 30
X_test = []
y_true = []
for i in range(0, len(data) - sequence_length):
    X_test.append(data[i:i + sequence_length])
    y_true.append(data[i + sequence_length])
X_test = np.array(X_test[:10])  
y_true = np.array(y_true[:10])
print("X_test shape:", X_test.shape)


y_pred = model.predict(X_test)
print("True x, y:\n", y_true)
print("Predicted x, y:\n", y_pred)