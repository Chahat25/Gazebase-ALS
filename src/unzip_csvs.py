import os
import zipfile
import glob
import time

# Paths
data_dir =r"C:\Users\Chahat\eye_communicator\data"
csv_dir =r"C:\Users\Chahat\eye_communicator\data\csvs"
os.makedirs(csv_dir, exist_ok=True)

# Debug
print("Data directory:", data_dir)
print("CSV output directory:", csv_dir)

# Find Round_* folders
round_folders = sorted([f for f in os.listdir(data_dir) if f.startswith("Round_") and os.path.isdir(os.path.join(data_dir, f))])
if not round_folders:
    print("Error: No Round_* folders found in", data_dir)
    print("Available:", os.listdir(data_dir))
    exit()

# Unzip all .zip files
total_extracted = 0
total_csvs_expected = 0
for round_folder in round_folders:
    round_path = os.path.join(data_dir, round_folder)
    print(f"Processing: {round_folder}")
    zip_files = sorted([f for f in os.listdir(round_path) if f.endswith(".zip")])
    if not zip_files:
        print(f"No .zip files in {round_folder}")
        continue

    for zip_file in zip_files:
        zip_path = os.path.join(round_path, zip_file)
        subject_id = os.path.splitext(zip_file)[0]  # e.g., Subject_1001
        subject_dir = os.path.join(csv_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        print(f"Unzipping: {zip_file} to {subject_dir}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_count = len([f for f in zip_ref.namelist() if f.endswith(".csv")])
                if csv_count == 0:
                    print(f"Warning: No CSVs in {zip_file}")
                total_csvs_expected += csv_count
                zip_ref.extractall(subject_dir)
            print(f"Extracted: {zip_file} ({csv_count} CSVs)")
            total_extracted += 1
        except Exception as e:
            print(f"Error unzipping {zip_file}: {e}")

# Verify
time.sleep(2)
csv_files = glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
print("All CSVs extracted to:", csv_dir)
print("Total .zip files extracted:", total_extracted)
print("Total CSVs expected:", total_csvs_expected)
print("Total CSVs found:", len(csv_files))
if csv_files:
    print("Sample CSVs:", [os.path.relpath(f, csv_dir) for f in csv_files[:5]])