import pandas as pd
import os
import glob

# Paths
data_dir = r"C:\Users\Chahat\eye_communicator\data\csvs"
output_dir =r"C:\Users\Chahat\eye_communicator\data\processed"
os.makedirs(output_dir, exist_ok=True)

# Find CSVs
csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
if not csv_files:
    print(f"Error: No CSVs in {data_dir}")
    print("Available files:", os.listdir(data_dir))
    exit()

print(f"Found {len(csv_files)} CSVs")

# Limit for speed
max_files = 1000
csv_files = csv_files[:max_files]
print(f"Processing {len(csv_files)} CSVs")

# Process
processed = 0
for idx, sample_csv in enumerate(csv_files):
    sample_name = os.path.basename(sample_csv)
    print(f"Processing {idx + 1}/{len(csv_files)}: {sample_name}")

    try:
        df = pd.read_csv(sample_csv, encoding='utf-8', delimiter=',')
    except Exception as e:
        print(f"Error reading {sample_name}: {e}")
        continue

    gaze_cols = ['x', 'y'] if 'x' in df.columns and 'y' in df.columns else None
    if not gaze_cols:
        print(f"No gaze columns in {sample_name}. Skipping.")
        continue

    df_clean = df.dropna(subset=gaze_cols)[['x', 'y']]
    print(f"Rows: Original={len(df)}, Cleaned={len(df_clean)}")

    output_file = os.path.join(output_dir, f"clean_{sample_name}")
    df_clean.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    processed += 1

print(f"Processed {processed} CSVs")