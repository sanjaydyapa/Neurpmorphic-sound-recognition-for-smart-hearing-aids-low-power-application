import soundata
import os

print("Initializing dataset...")
dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')

print("\nChecking what files soundata expects...")
print(f"Data home: {dataset.data_home}")
print(f"\nDataset index info:")
print(dataset._index)

print("\n\nLet's check what files are actually in the data directory:")
data_home = 'urbansound8k_data'
for root, dirs, files in os.walk(data_home):
    level = root.replace(data_home, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... and {len(files) - 5} more files')
