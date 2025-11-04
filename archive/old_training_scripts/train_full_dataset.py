"""
ADVANCED TRAINING - Use Entire Dataset
Train with ALL available clips instead of just 20
"""

import soundata
import librosa
import numpy as np
from neuromorphic_sound_detector_final import NeuromorphicDetector, create_fingerprint
import matplotlib.pyplot as plt
import json
import time

def train_with_full_dataset(target_class='siren', use_all_clips=True, max_clips=None):
    """
    Train detector using the entire dataset or a specified number of clips
    
    Args:
        target_class: Sound class to train on ('siren', 'dog_bark', etc.)
        use_all_clips: If True, use all available clips; if False, use max_clips
        max_clips: Number of clips to use (ignored if use_all_clips=True)
    """
    print("=" * 70)
    print("TRAINING WITH FULL DATASET")
    print("=" * 70)
    print()
    
    # Load dataset
    print("📥 Loading UrbanSound8K dataset...")
    dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
    
    # Get all clips of target class
    all_clips = [clip for clip in dataset.load_clips().values() 
                 if clip.class_label == target_class]
    
    total_available = len(all_clips)
    print(f"✅ Found {total_available} clips of class '{target_class}'")
    
    # Determine how many to use
    if use_all_clips:
        training_clips = all_clips
        num_clips = len(training_clips)
        print(f"🎯 Using ALL {num_clips} clips for training")
    else:
        num_clips = min(max_clips if max_clips else 20, total_available)
        training_clips = all_clips[:num_clips]
        print(f"🎯 Using {num_clips} clips for training")
    
    print()
    print(f"--- Building fingerprint from {num_clips} clips ---")
    print("This may take a few minutes for large datasets...")
    print()
    
    # Extract fingerprints from all training clips
    fingerprints = []
    start_time = time.time()
    
    for i, clip in enumerate(training_clips, 1):
        try:
            # Load audio
            audio, sr = librosa.load(clip.audio_path, sr=22050)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Average over time to get fingerprint
            fingerprint = np.mean(mfccs, axis=1)
            fingerprints.append(fingerprint)
            
            # Progress indicator
            if i % 10 == 0 or i == num_clips:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (num_clips - i) / rate if rate > 0 else 0
                print(f"  Progress: {i}/{num_clips} clips ({i/num_clips*100:.1f}%) "
                      f"- ETA: {eta:.0f}s", end='\r')
        
        except Exception as e:
            print(f"\n  ⚠️  Error processing clip {i}: {e}")
            continue
    
    print()  # New line after progress
    
    # Calculate master fingerprint (average of all)
    target_fingerprint = np.mean(fingerprints, axis=0)
    
    training_time = time.time() - start_time
    print(f"✅ Fingerprint built from {len(fingerprints)} clips in {training_time:.2f}s")
    
    # Calculate statistics about the fingerprints
    fingerprints_array = np.array(fingerprints)
    std_devs = np.std(fingerprints_array, axis=0)
    
    print()
    print("📊 Training Statistics:")
    print(f"  Mean MFCC values: {np.mean(target_fingerprint):.4f}")
    print(f"  Std deviation: {np.mean(std_devs):.4f}")
    print(f"  Min MFCC: {np.min(target_fingerprint):.4f}")
    print(f"  Max MFCC: {np.max(target_fingerprint):.4f}")
    
    # Save fingerprint to file
    fingerprint_data = {
        'target_class': target_class,
        'num_training_clips': len(fingerprints),
        'fingerprint': target_fingerprint.tolist(),
        'std_devs': std_devs.tolist(),
        'training_time': training_time
    }
    
    filename = f'fingerprint_{target_class}_{len(fingerprints)}clips.json'
    with open(filename, 'w') as f:
        json.dump(fingerprint_data, f, indent=2)
    
    print(f"💾 Fingerprint saved to: {filename}")
    
    return target_fingerprint, fingerprints_array


def load_fingerprint_from_file(filename):
    """Load a previously saved fingerprint"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return np.array(data['fingerprint']), data


def test_with_full_training():
    """
    Complete example: Train with full dataset and test
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: TRAIN WITH FULL DATASET AND TEST")
    print("=" * 70)
    print()
    
    # Get user input
    print("Available sound classes:")
    print("  1. siren")
    print("  2. dog_bark")
    print("  3. street_music")
    print("  4. car_horn")
    print("  5. baby_crying")
    print("  6. drilling")
    print("  7. engine_idling")
    print("  8. gun_shot")
    print("  9. jackhammer")
    print(" 10. air_conditioner")
    print()
    
    target_class = input("Enter target class name (default: siren): ").strip() or 'siren'
    
    print()
    print("Training options:")
    print("  1. Use ALL available clips (recommended for best accuracy)")
    print("  2. Use specific number of clips")
    print()
    
    choice = input("Enter choice (1 or 2, default: 1): ").strip() or '1'
    
    if choice == '1':
        # Train with all clips
        target_fingerprint, all_fingerprints = train_with_full_dataset(
            target_class=target_class,
            use_all_clips=True
        )
    else:
        # Train with specific number
        num = input("Enter number of clips (default: 100): ").strip()
        num_clips = int(num) if num else 100
        target_fingerprint, all_fingerprints = train_with_full_dataset(
            target_class=target_class,
            use_all_clips=False,
            max_clips=num_clips
        )
    
    # Initialize detector with optimized thresholds
    print()
    print("=" * 70)
    print("INITIALIZING DETECTOR")
    print("=" * 70)
    
    ENERGY_THRESHOLD = 0.05
    MATCH_THRESHOLD = 4500
    
    detector = NeuromorphicDetector(
        target_fingerprint,
        ENERGY_THRESHOLD,
        MATCH_THRESHOLD,
        target_class
    )
    
    # Test on some clips
    print()
    print("=" * 70)
    print("TESTING DETECTOR")
    print("=" * 70)
    print()
    
    dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
    
    # Get test clips (clips NOT used in training)
    all_target_clips = [clip for clip in dataset.load_clips().values() 
                        if clip.class_label == target_class]
    
    # Use last 10 clips for testing (assuming we trained on first N)
    test_clips = all_target_clips[-10:]
    
    # Also get some negative examples
    negative_classes = ['street_music', 'dog_bark', 'car_horn']
    negative_clips = []
    for neg_class in negative_classes:
        clips = [c for c in dataset.load_clips().values() if c.class_label == neg_class]
        if clips:
            negative_clips.append(clips[0])
    
    # Evaluate
    tp, fp, tn, fn = 0, 0, 0, 0
    
    print(f"Testing on {len(test_clips)} positive + {len(negative_clips)} negative clips...")
    print()
    
    for clip in test_clips:
        audio, sr = librosa.load(clip.audio_path, sr=22050)
        chunk_size = int(0.25 * sr)
        
        detected = False
        for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[i:i + chunk_size]
            if detector.process_chunk(chunk, sr, i // chunk_size):
                detected = True
                break
        
        if detected:
            tp += 1
            print(f"  ✅ {clip.clip_id}: CORRECTLY DETECTED")
        else:
            fn += 1
            print(f"  ❌ {clip.clip_id}: MISSED")
    
    for clip in negative_clips:
        audio, sr = librosa.load(clip.audio_path, sr=22050)
        chunk_size = int(0.25 * sr)
        
        detected = False
        for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[i:i + chunk_size]
            if detector.process_chunk(chunk, sr, i // chunk_size):
                detected = True
                break
        
        if detected:
            fp += 1
            print(f"  ❌ {clip.clip_id} ({clip.class_label}): FALSE POSITIVE")
        else:
            tn += 1
            print(f"  ✅ {clip.clip_id} ({clip.class_label}): CORRECTLY IGNORED")
    
    # Calculate metrics
    print()
    print("=" * 70)
    print("RESULTS WITH FULL DATASET TRAINING")
    print("=" * 70)
    print()
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print()
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print()
    
    print("💡 Compare these results with the 20-clip training:")
    print("   - More training data usually improves generalization")
    print("   - Watch for diminishing returns beyond ~100 clips")
    print("   - Balance training time vs accuracy gains")
    print()


def compare_training_sizes():
    """
    Compare detector performance with different training set sizes
    """
    print("\n" + "=" * 70)
    print("COMPARING TRAINING SET SIZES")
    print("=" * 70)
    print()
    
    target_class = 'siren'
    training_sizes = [10, 20, 50, 100, 200, 'all']
    
    results = []
    
    for size in training_sizes:
        print(f"\n--- Training with {size} clips ---")
        
        if size == 'all':
            fingerprint, _ = train_with_full_dataset(target_class, use_all_clips=True)
        else:
            fingerprint, _ = train_with_full_dataset(target_class, use_all_clips=False, max_clips=size)
        
        # Quick test (simplified)
        detector = NeuromorphicDetector(fingerprint, 0.05, 4500, target_class)
        
        # Test on 5 clips
        dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
        test_clips = [c for c in dataset.load_clips().values() 
                     if c.class_label == target_class][-5:]
        
        correct = 0
        for clip in test_clips:
            audio, sr = librosa.load(clip.audio_path, sr=22050)
            chunk_size = int(0.25 * sr)
            
            detected = False
            for i in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[i:i + chunk_size]
                if detector.process_chunk(chunk, sr, i // chunk_size):
                    detected = True
                    break
            
            if detected:
                correct += 1
        
        accuracy = correct / len(test_clips) * 100
        results.append({'size': size, 'accuracy': accuracy})
        print(f"  Accuracy: {accuracy:.1f}%")
    
    # Plot results
    print("\n📊 Plotting training size vs accuracy...")
    
    sizes = [r['size'] if r['size'] != 'all' else 300 for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training Set Size vs Detection Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=70, color='r', linestyle='--', label='Target: 70%')
    plt.legend()
    
    # Annotate points
    for i, (size, acc) in enumerate(zip(sizes, accuracies)):
        label = 'all' if results[i]['size'] == 'all' else str(results[i]['size'])
        plt.annotate(f'{label}\n{acc:.1f}%', 
                    (size, acc), 
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig('training_size_comparison.png', dpi=300)
    print("✅ Saved: training_size_comparison.png")
    plt.close()
    
    print()
    print("💡 Insights:")
    print("  - More training data generally improves accuracy")
    print("  - Diminishing returns typically occur after 100-200 clips")
    print("  - Balance training time vs accuracy gains")


if __name__ == "__main__":
    print("\n🎯 ADVANCED TRAINING OPTIONS\n")
    print("1. Train with full dataset and test")
    print("2. Compare different training sizes")
    print("3. Quick train with custom number of clips")
    print()
    
    choice = input("Enter choice (1-3, default: 1): ").strip() or '1'
    
    if choice == '1':
        test_with_full_training()
    elif choice == '2':
        compare_training_sizes()
    elif choice == '3':
        target_class = input("Target class (default: siren): ").strip() or 'siren'
        num = input("Number of clips (default: 100): ").strip()
        num_clips = int(num) if num else 100
        
        fingerprint, _ = train_with_full_dataset(
            target_class=target_class,
            use_all_clips=False,
            max_clips=num_clips
        )
        
        print("\n✅ Training complete!")
        print(f"   Fingerprint saved to: fingerprint_{target_class}_{num_clips}clips.json")
