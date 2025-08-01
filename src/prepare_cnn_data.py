import pandas as pd
import numpy as np
import os
import glob

# Paths
data_dir =r"C:\Users\Chahat\eye_communicator\data\processed"
output_dir =r"C:\Users\Chahat\eye_communicator\data\cnn"
os.makedirs(output_dir, exist_ok=True)

# Parameters
sequence_length = 30  # Frames per sequence
stride = 10  # Step between sequences
max_files = 500  # Limit CSVs for memory

# Load CSVs
csv_files = glob.glob(os.path.join(data_dir, "clean_*.csv"))
if not csv_files:
    print(f"Error: No clean CSVs in {data_dir}")
    exit()

csv_files = csv_files[:max_files]
print(f"Processing {len(csv_files)} CSVs")

X, y = [], []
for idx, csv_file in enumerate(csv_files):
    print(f"Loading {idx + 1}/{len(csv_files)}: {os.path.basename(csv_file)}")
    try:
        df = pd.read_csv(csv_file, usecols=['x', 'y'])
        data = df[['x', 'y']].values
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)  # Normalize
        for i in range(0, len(data) - sequence_length, stride):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length - 1])
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

X = np.array(X)
y = np.array(y)
print("X shape:", X.shape)
print("y shape:", y.shape)

np.save(os.path.join(output_dir, "X.npy"), X)
np.save(os.path.join(output_dir, "y.npy"), y)
print("Saved CNN data to:", output_dir)