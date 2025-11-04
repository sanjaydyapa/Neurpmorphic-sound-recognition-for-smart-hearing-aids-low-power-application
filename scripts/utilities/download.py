import soundata

print("Initializing dataset...")
# This 'data_home' must match the one in your project.py
dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data') 

print("Starting dataset download...")
print("This is a large file (around 6 GB) and will take a while. Please be patient.")

# This command downloads all the audio files
dataset.download(force_overwrite=True)
print("\n--- DOWNLOAD COMPLETE! ---")
print("You can now run your main 'project.py' script.")