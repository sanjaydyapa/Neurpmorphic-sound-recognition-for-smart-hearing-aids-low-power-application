"""
MULTI-CLASS NEUROMORPHIC DETECTOR
Train on ALL sound classes in UrbanSound8K dataset
Detect and classify all 10 urban sound categories
"""

import soundata
import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# All 10 sound classes in UrbanSound8K
SOUND_CLASSES = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music'
]


class MultiClassDetector:
    """
    Neuromorphic detector that can recognize ALL sound classes
    Uses multiple fingerprints (one per class) and finds best match
    """
    def __init__(self, class_fingerprints, energy_threshold=0.05):
        """
        Args:
            class_fingerprints: Dictionary mapping class names to their fingerprints
            energy_threshold: Minimum energy to process a chunk
        """
        self.class_fingerprints = class_fingerprints
        self.energy_threshold = energy_threshold
        self.sound_classes = list(class_fingerprints.keys())
        
        print("\n╔════════════════════════════════════════════════════════════╗")
        print("║      MULTI-CLASS NEUROMORPHIC DETECTOR INITIALIZED        ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"  Trained Classes: {len(self.sound_classes)}")
        for cls in self.sound_classes:
            print(f"    • {cls}")
        print(f"  Energy Threshold: {self.energy_threshold}")
        print()
    
    def process_chunk(self, chunk, sr):
        """
        Process audio chunk and classify it
        Returns: (detected_class, confidence) or (None, 0) if no detection
        """
        # Stage 1: Energy gate (like before)
        energy = np.sqrt(np.mean(chunk**2))
        
        if energy < self.energy_threshold:
            return None, 0  # Too quiet, skip
        
        # Stage 2: Extract MFCCs
        mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        chunk_fingerprint = np.mean(mfccs, axis=1)
        
        # Stage 3: Compare with ALL class fingerprints
        best_match_class = None
        best_match_error = float('inf')
        
        for sound_class, target_fingerprint in self.class_fingerprints.items():
            # Calculate MSE
            error = np.mean((chunk_fingerprint - target_fingerprint) ** 2)
            
            if error < best_match_error:
                best_match_error = error
                best_match_class = sound_class
        
        # Calculate confidence (lower error = higher confidence)
        # Normalize error to confidence score (0-100%)
        confidence = max(0, 100 - (best_match_error / 100))
        
        return best_match_class, confidence
    
    def classify_audio(self, audio_path, min_confidence=30):
        """
        Classify entire audio file
        Returns: (predicted_class, confidence, detection_count)
        """
        audio, sr = librosa.load(audio_path, sr=22050)
        chunk_size = int(0.25 * sr)
        
        # Track detections per class
        class_votes = {cls: 0 for cls in self.sound_classes}
        total_chunks = 0
        
        for i in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[i:i + chunk_size]
            detected_class, confidence = self.process_chunk(chunk, sr)
            
            if detected_class and confidence >= min_confidence:
                class_votes[detected_class] += 1
                total_chunks += 1
        
        # Winner takes all - class with most votes
        if total_chunks == 0:
            return None, 0, 0
        
        predicted_class = max(class_votes, key=class_votes.get)
        class_confidence = (class_votes[predicted_class] / total_chunks) * 100
        
        return predicted_class, class_confidence, class_votes[predicted_class]


def extract_fingerprint_for_class(dataset, sound_class, num_clips=None, train_ratio=0.8):
    """
    Extract fingerprint for a specific sound class
    Uses train/test split for proper evaluation
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING CLASS: {sound_class.upper()}")
    print(f"{'='*80}")
    
    # Get all clips for this class
    all_clips = [clip for clip in dataset.load_clips().values() 
                 if clip.tags.labels[0] == sound_class]
    
    print(f"Found {len(all_clips)} clips")
    
    # Shuffle and split
    np.random.shuffle(all_clips)
    
    if num_clips:
        all_clips = all_clips[:num_clips]
    
    split_idx = int(len(all_clips) * train_ratio)
    train_clips = all_clips[:split_idx]
    test_clips = all_clips[split_idx:]
    
    print(f"Training: {len(train_clips)} clips")
    print(f"Testing: {len(test_clips)} clips")
    
    # Extract fingerprints from training clips
    fingerprints = []
    start_time = time.time()
    
    for i, clip in enumerate(train_clips, 1):
        try:
            audio, sr = librosa.load(clip.audio_path, sr=22050)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            fingerprint = np.mean(mfccs, axis=1)
            fingerprints.append(fingerprint)
            
            if i % 50 == 0 or i == len(train_clips):
                print(f"  Progress: {i}/{len(train_clips)} ({i/len(train_clips)*100:.1f}%)", end='\r')
        except Exception as e:
            continue
    
    print()  # Newline
    
    # Calculate master fingerprint
    target_fingerprint = np.mean(fingerprints, axis=0)
    
    training_time = time.time() - start_time
    print(f"✅ Fingerprint extracted in {training_time:.2f}s")
    print(f"  Mean MFCC: {np.mean(target_fingerprint):.4f}")
    print(f"  Std: {np.std(target_fingerprint):.4f}")
    
    return target_fingerprint, train_clips, test_clips


def train_all_classes(max_clips_per_class=None):
    """
    Train detector on ALL sound classes
    """
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*15 + "MULTI-CLASS TRAINING - ALL SOUNDS" + " "*28 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    # Load dataset
    print("📥 Loading UrbanSound8K dataset...")
    dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
    print("✅ Dataset loaded")
    
    # Train on each class
    class_fingerprints = {}
    all_train_clips = {}
    all_test_clips = {}
    
    total_start_time = time.time()
    
    for sound_class in SOUND_CLASSES:
        fingerprint, train_clips, test_clips = extract_fingerprint_for_class(
            dataset, 
            sound_class,
            num_clips=max_clips_per_class
        )
        
        class_fingerprints[sound_class] = fingerprint
        all_train_clips[sound_class] = train_clips
        all_test_clips[sound_class] = test_clips
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("✅ ALL CLASSES TRAINED")
    print(f"{'='*80}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Classes trained: {len(class_fingerprints)}")
    print()
    
    return class_fingerprints, all_train_clips, all_test_clips


def evaluate_multi_class(detector, all_test_clips):
    """
    Comprehensive evaluation on all classes
    """
    print("\n" + "="*80)
    print("EVALUATING MULTI-CLASS DETECTOR")
    print("="*80)
    print()
    
    y_true = []
    y_pred = []
    
    total_clips = sum(len(clips) for clips in all_test_clips.values())
    processed = 0
    
    print(f"Testing on {total_clips} clips across {len(all_test_clips)} classes...")
    print()
    
    for true_class, test_clips in all_test_clips.items():
        print(f"\nTesting class: {true_class}")
        print(f"{'─'*80}")
        
        for i, clip in enumerate(test_clips, 1):
            try:
                predicted_class, confidence, votes = detector.classify_audio(
                    clip.audio_path,
                    min_confidence=30
                )
                
                if predicted_class is None:
                    predicted_class = 'unknown'
                
                y_true.append(true_class)
                y_pred.append(predicted_class)
                
                status = "✅" if predicted_class == true_class else "❌"
                print(f"  [{i}/{len(test_clips)}] {status} True: {true_class:20s} | "
                      f"Predicted: {predicted_class:20s} | Confidence: {confidence:.1f}%")
                
                processed += 1
                
            except Exception as e:
                print(f"  [{i}/{len(test_clips)}] ⚠️  Error: {e}")
                continue
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("RESULTS - MULTI-CLASS CLASSIFICATION")
    print(f"{'='*80}")
    print()
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"🎯 OVERALL ACCURACY: {accuracy*100:.2f}%")
    print()
    
    # Per-class metrics
    print("📊 CLASSIFICATION REPORT:")
    print()
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=SOUND_CLASSES)
    
    # Visualize confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=SOUND_CLASSES,
                yticklabels=SOUND_CLASSES,
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Multi-Class Confusion Matrix\nAll 10 UrbanSound8K Categories', 
             fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_all_classes.png', dpi=300, bbox_inches='tight')
    print("\n📊 Saved: confusion_matrix_all_classes.png")
    plt.close()
    
    # Per-class accuracy bar chart
    class_accuracies = []
    for cls in SOUND_CLASSES:
        cls_true = [i for i, t in enumerate(y_true) if t == cls]
        if cls_true:
            cls_correct = sum(1 for i in cls_true if y_pred[i] == cls)
            cls_acc = cls_correct / len(cls_true) * 100
            class_accuracies.append(cls_acc)
        else:
            class_accuracies.append(0)
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(range(len(SOUND_CLASSES)), class_accuracies, color='steelblue', edgecolor='black')
    plt.xlabel('Sound Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Detection Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(range(len(SOUND_CLASSES)), SOUND_CLASSES, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Add average line
    avg_acc = np.mean(class_accuracies)
    plt.axhline(avg_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_acc:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: per_class_accuracy.png")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'class_accuracies': dict(zip(SOUND_CLASSES, class_accuracies)),
        'y_true': y_true,
        'y_pred': y_pred
    }


def save_multi_class_model(class_fingerprints, metrics):
    """
    Save the complete multi-class model
    """
    model_data = {
        'sound_classes': SOUND_CLASSES,
        'num_classes': len(SOUND_CLASSES),
        'fingerprints': {
            cls: fp.tolist() for cls, fp in class_fingerprints.items()
        },
        'overall_accuracy': float(metrics['accuracy']),
        'class_accuracies': metrics['class_accuracies'],
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'multi_class_neuromorphic_detector'
    }
    
    filename = 'multi_class_model_all_sounds.json'
    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"\n💾 Multi-class model saved to: {filename}")
    print(f"   • Trained on {len(SOUND_CLASSES)} classes")
    print(f"   • Overall accuracy: {metrics['accuracy']*100:.2f}%")
    print()
    
    return filename


def interactive_test(detector):
    """
    Interactive testing - classify any audio file
    """
    print("\n" + "="*80)
    print("INTERACTIVE TESTING - Classify Any Audio")
    print("="*80)
    print()
    
    dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
    all_clips = list(dataset.load_clips().values())
    
    while True:
        print("\nOptions:")
        print("  1. Test a random clip")
        print("  2. Test specific class")
        print("  3. Exit")
        
        choice = input("\nChoice (1-3): ").strip()
        
        if choice == '3':
            break
        elif choice == '1':
            clip = np.random.choice(all_clips)
        elif choice == '2':
            print("\nAvailable classes:")
            for i, cls in enumerate(SOUND_CLASSES, 1):
                print(f"  {i}. {cls}")
            
            cls_idx = int(input("\nSelect class (1-10): ")) - 1
            if 0 <= cls_idx < len(SOUND_CLASSES):
                target_class = SOUND_CLASSES[cls_idx]
                matching_clips = [c for c in all_clips if c.tags.labels[0] == target_class]
                if matching_clips:
                    clip = np.random.choice(matching_clips)
                else:
                    print("No clips found!")
                    continue
            else:
                print("Invalid choice!")
                continue
        else:
            print("Invalid choice!")
            continue
        
        # Classify
        print(f"\n🔊 Testing: {clip.clip_id}")
        print(f"   True class: {clip.tags.labels[0]}")
        
        predicted_class, confidence, votes = detector.classify_audio(clip.audio_path)
        
        print(f"\n   Predicted: {predicted_class}")
        print(f"   Confidence: {confidence:.1f}%")
        print(f"   Detection votes: {votes}")
        
        if predicted_class == clip.tags.labels[0]:
            print("   ✅ CORRECT!")
        else:
            print("   ❌ WRONG!")


def main():
    """
    Complete multi-class training pipeline
    """
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*10 + "MULTI-CLASS NEUROMORPHIC DETECTOR TRAINING" + " "*25 + "║")
    print("║" + " "*18 + "Train on ALL 10 Sound Classes" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    print("This will train a detector for ALL 10 sound classes:")
    for i, cls in enumerate(SOUND_CLASSES, 1):
        print(f"  {i:2d}. {cls}")
    print()
    
    # Ask user for training options
    print("Training options:")
    print("  1. Use ALL available clips (best accuracy, ~30-45 minutes)")
    print("  2. Use limited clips per class (faster, ~10-15 minutes)")
    print()
    
    choice = input("Choice (1 or 2, default: 1): ").strip() or '1'
    
    if choice == '2':
        num_clips = input("Max clips per class (default: 200): ").strip()
        max_clips = int(num_clips) if num_clips else 200
    else:
        max_clips = None
    
    print()
    
    # Step 1: Train all classes
    class_fingerprints, all_train_clips, all_test_clips = train_all_classes(max_clips)
    
    # Step 2: Create detector
    print("\n" + "="*80)
    print("CREATING MULTI-CLASS DETECTOR")
    print("="*80)
    detector = MultiClassDetector(class_fingerprints, energy_threshold=0.05)
    
    # Step 3: Evaluate
    metrics = evaluate_multi_class(detector, all_test_clips)
    
    # Step 4: Save model
    model_file = save_multi_class_model(class_fingerprints, metrics)
    
    # Summary
    print("\n" + "="*80)
    print("✅ MULTI-CLASS TRAINING COMPLETE!")
    print("="*80)
    print()
    print("📁 Generated files:")
    print(f"  • {model_file} - Complete multi-class model")
    print("  • confusion_matrix_all_classes.png - 10x10 confusion matrix")
    print("  • per_class_accuracy.png - Per-class performance")
    print()
    print(f"🎯 OVERALL ACCURACY: {metrics['accuracy']*100:.2f}%")
    print()
    print("💡 You can now:")
    print("  • Detect and classify ANY urban sound")
    print("  • Use this model for real-time classification")
    print("  • Test interactively below")
    print()
    
    # Optional interactive testing
    test_more = input("Test detector interactively? (y/n): ").strip().lower()
    if test_more == 'y':
        interactive_test(detector)
    
    print("\n🎉 Done! Your multi-class detector is ready!")


if __name__ == "__main__":
    main()
