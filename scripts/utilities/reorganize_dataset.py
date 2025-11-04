import os
import shutil

print("Reorganizing UrbanSound8K dataset for soundata...")

data_home = 'urbansound8k_data'

# Step 1: Move fold folders into audio/ directory
print("\n1. Creating audio/ directory structure...")
audio_dir = os.path.join(data_home, 'audio')
os.makedirs(audio_dir, exist_ok=True)

for i in range(1, 11):
    old_fold = os.path.join(data_home, f'fold{i}')
    new_fold = os.path.join(audio_dir, f'fold{i}')
    
    if os.path.exists(old_fold) and not os.path.exists(new_fold):
        print(f"  Moving fold{i}/ -> audio/fold{i}/")
        shutil.move(old_fold, new_fold)
    elif os.path.exists(new_fold):
        print(f"  audio/fold{i}/ already exists, skipping...")

# Step 2: Move UrbanSound8K.csv into metadata/ directory
print("\n2. Creating metadata/ directory structure...")
metadata_dir = os.path.join(data_home, 'metadata')
os.makedirs(metadata_dir, exist_ok=True)

csv_file = os.path.join(data_home, 'UrbanSound8K.csv')
new_csv = os.path.join(metadata_dir, 'UrbanSound8K.csv')

if os.path.exists(csv_file) and not os.path.exists(new_csv):
    print(f"  Moving UrbanSound8K.csv -> metadata/UrbanSound8K.csv")
    shutil.move(csv_file, new_csv)
elif os.path.exists(new_csv):
    print(f"  metadata/UrbanSound8K.csv already exists, skipping...")

print("\n--- Reorganization Complete! ---")
print("\nYour directory structure should now look like:")
print("urbansound8k_data/")
print("  audio/")
print("    fold1/")
print("    fold2/")
print("    ...")
print("  metadata/")
print("    UrbanSound8K.csv")
print("\nNow your project.py should work without needing to download anything!")
