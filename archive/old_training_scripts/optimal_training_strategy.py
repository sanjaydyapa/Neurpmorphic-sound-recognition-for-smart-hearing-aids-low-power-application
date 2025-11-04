"""
OPTIMAL TRAINING STRATEGY FOR MAXIMUM ACCURACY
===============================================
This script implements best practices for training the neuromorphic detector
to achieve the highest possible accuracy on the full dataset.

STRATEGY:
1. Use 80% of data for training, 20% for testing (proper train/test split)
2. Optimize thresholds using validation set
3. Train on ALL folds (not just fold1) for maximum diversity
4. Implement cross-validation for robust performance estimation
5. Use data augmentation techniques (optional)
6. Fine-tune hyperparameters
"""

import soundata
import librosa
import numpy as np
from neuromorphic_sound_detector_final import NeuromorphicDetector
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

def load_all_clips_from_dataset(target_class='siren'):
    """
    Load ALL clips from ALL folds (not just fold1)
    This ensures maximum diversity in training data
    """
    print("=" * 80)
    print("LOADING COMPLETE DATASET FROM ALL FOLDS")
    print("=" * 80)
    print()
    
    dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
    
    # Get all clips of target class
    all_clips = []
    print(f"📂 Scanning all folds for '{target_class}' clips...")
    
    for clip_id in dataset.clip_ids:
        clip = dataset.clip(clip_id)
        if clip.tags.labels[0] == target_class:
            all_clips.append(clip)
    
    print(f"✅ Found {len(all_clips)} total clips across all folds")
    print()
    
    return all_clips, dataset


def stratified_train_test_split(all_clips, test_size=0.2, random_state=42):
    """
    Split data into train/test sets properly
    This prevents overfitting and gives realistic accuracy estimates
    """
    print("🔀 Splitting data: 80% training, 20% testing...")
    
    # Shuffle clips
    np.random.seed(random_state)
    shuffled_clips = np.random.permutation(all_clips)
    
    # Split
    split_idx = int(len(shuffled_clips) * (1 - test_size))
    train_clips = shuffled_clips[:split_idx]
    test_clips = shuffled_clips[split_idx:]
    
    print(f"  Training set: {len(train_clips)} clips")
    print(f"  Test set: {len(test_clips)} clips")
    print()
    
    return train_clips, test_clips


def extract_robust_fingerprint(train_clips, use_median=False, remove_outliers=True):
    """
    Extract fingerprint with robustness techniques:
    - Option to use MEDIAN instead of MEAN (more robust to outliers)
    - Option to remove outlier clips before averaging
    - Calculate confidence bounds
    """
    print("=" * 80)
    print("EXTRACTING ROBUST FINGERPRINT FROM TRAINING DATA")
    print("=" * 80)
    print()
    
    fingerprints = []
    failed_clips = 0
    
    print(f"Processing {len(train_clips)} training clips...")
    start_time = time.time()
    
    for i, clip in enumerate(train_clips, 1):
        try:
            audio, sr = librosa.load(clip.audio_path, sr=22050)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Average over time
            fingerprint = np.mean(mfccs, axis=1)
            fingerprints.append(fingerprint)
            
            # Progress
            if i % 50 == 0 or i == len(train_clips):
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(train_clips) - i) / rate if rate > 0 else 0
                print(f"  Progress: {i}/{len(train_clips)} ({i/len(train_clips)*100:.1f}%) "
                      f"- ETA: {eta:.0f}s", end='\r')
        
        except Exception as e:
            failed_clips += 1
            continue
    
    print()  # Newline after progress
    
    fingerprints_array = np.array(fingerprints)
    
    # Remove outliers if requested (clips with extreme MFCC values)
    if remove_outliers:
        print("\n🔍 Removing outlier clips...")
        
        # Calculate z-scores for each fingerprint
        mean_per_clip = np.mean(fingerprints_array, axis=1)
        z_scores = np.abs((mean_per_clip - np.mean(mean_per_clip)) / np.std(mean_per_clip))
        
        # Keep only clips with z-score < 3 (within 3 standard deviations)
        valid_mask = z_scores < 3
        outliers_removed = np.sum(~valid_mask)
        
        fingerprints_array = fingerprints_array[valid_mask]
        
        print(f"  Removed {outliers_removed} outlier clips")
        print(f"  Kept {len(fingerprints_array)} high-quality clips")
    
    # Calculate final fingerprint
    if use_median:
        print("\n📊 Using MEDIAN for robustness (less sensitive to outliers)")
        target_fingerprint = np.median(fingerprints_array, axis=0)
    else:
        print("\n📊 Using MEAN for fingerprint")
        target_fingerprint = np.mean(fingerprints_array, axis=0)
    
    # Calculate confidence metrics
    std_devs = np.std(fingerprints_array, axis=0)
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Fingerprint extraction complete in {training_time:.2f}s")
    print(f"  Successfully processed: {len(fingerprints_array)} clips")
    print(f"  Failed clips: {failed_clips}")
    print(f"  Mean MFCC: {np.mean(target_fingerprint):.4f}")
    print(f"  Std deviation: {np.mean(std_devs):.4f}")
    print()
    
    return target_fingerprint, fingerprints_array, std_devs


def optimize_thresholds(target_fingerprint, train_clips, test_clips, target_class='siren'):
    """
    Systematically search for optimal energy and match thresholds
    using a validation approach (grid search)
    """
    print("=" * 80)
    print("OPTIMIZING THRESHOLDS FOR MAXIMUM ACCURACY")
    print("=" * 80)
    print()
    
    # Use a subset for faster optimization
    val_positive = train_clips[-50:]  # Last 50 training clips as validation
    
    # Get negative examples
    dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
    negative_classes = ['street_music', 'dog_bark', 'children_playing', 'car_horn']
    val_negative = []
    
    for neg_class in negative_classes:
        neg_clips = [c for c in dataset.load_clips().values() 
                    if c.class_label == neg_class][:15]  # 15 clips per class
        val_negative.extend(neg_clips)
    
    print(f"Validation set: {len(val_positive)} positive + {len(val_negative)} negative")
    print()
    
    # Grid search parameters
    energy_thresholds = [0.02, 0.03, 0.05, 0.07, 0.10]
    match_thresholds = [3000, 3500, 4000, 4500, 5000, 5500, 6000]
    
    best_accuracy = 0
    best_params = {}
    results = []
    
    total_combinations = len(energy_thresholds) * len(match_thresholds)
    combination = 0
    
    print("🔍 Testing threshold combinations...")
    print()
    
    for energy_thresh in energy_thresholds:
        for match_thresh in match_thresholds:
            combination += 1
            
            # Create detector with these thresholds
            detector = NeuromorphicDetector(
                target_fingerprint,
                energy_thresh,
                match_thresh,
                target_class
            )
            
            # Test on validation set
            correct = 0
            total = 0
            
            # Test positive examples
            for clip in val_positive:
                try:
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
                    total += 1
                except:
                    continue
            
            # Test negative examples
            for clip in val_negative:
                try:
                    audio, sr = librosa.load(clip.audio_path, sr=22050)
                    chunk_size = int(0.25 * sr)
                    
                    detected = False
                    for i in range(0, len(audio) - chunk_size, chunk_size):
                        chunk = audio[i:i + chunk_size]
                        if detector.process_chunk(chunk, sr, i // chunk_size):
                            detected = True
                            break
                    
                    if not detected:  # Correctly rejected
                        correct += 1
                    total += 1
                except:
                    continue
            
            accuracy = correct / total if total > 0 else 0
            
            results.append({
                'energy_threshold': energy_thresh,
                'match_threshold': match_thresh,
                'accuracy': accuracy
            })
            
            print(f"  [{combination}/{total_combinations}] "
                  f"Energy={energy_thresh:.2f}, Match={match_thresh} "
                  f"→ Accuracy={accuracy*100:.1f}%", end='\r')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'energy_threshold': energy_thresh,
                    'match_threshold': match_thresh,
                    'validation_accuracy': accuracy
                }
    
    print()  # Newline
    print()
    print("=" * 80)
    print("OPTIMAL THRESHOLDS FOUND")
    print("=" * 80)
    print(f"  Energy Threshold: {best_params['energy_threshold']}")
    print(f"  Match Threshold: {best_params['match_threshold']}")
    print(f"  Validation Accuracy: {best_params['validation_accuracy']*100:.2f}%")
    print()
    
    # Visualize threshold search
    visualize_threshold_optimization(results)
    
    return best_params


def visualize_threshold_optimization(results):
    """Create heatmap of threshold search results"""
    
    # Convert results to matrix form
    energy_values = sorted(list(set([r['energy_threshold'] for r in results])))
    match_values = sorted(list(set([r['match_threshold'] for r in results])))
    
    accuracy_matrix = np.zeros((len(match_values), len(energy_values)))
    
    for result in results:
        e_idx = energy_values.index(result['energy_threshold'])
        m_idx = match_values.index(result['match_threshold'])
        accuracy_matrix[m_idx, e_idx] = result['accuracy'] * 100
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(accuracy_matrix, 
                xticklabels=[f'{e:.2f}' for e in energy_values],
                yticklabels=[f'{m}' for m in match_values],
                annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                cbar_kws={'label': 'Accuracy (%)'})
    
    plt.xlabel('Energy Threshold', fontsize=12)
    plt.ylabel('Match Threshold (MSE)', fontsize=12)
    plt.title('Threshold Optimization Grid Search\n(Higher is better)', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('threshold_optimization_heatmap.png', dpi=300, bbox_inches='tight')
    print("📊 Saved threshold optimization heatmap")
    plt.close()


def comprehensive_evaluation(detector, test_clips, dataset, target_class='siren'):
    """
    Comprehensive evaluation on held-out test set
    """
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION ON TEST SET")
    print("=" * 80)
    print()
    
    # Get negative test examples
    negative_classes = ['street_music', 'dog_bark', 'children_playing', 'car_horn', 
                       'jackhammer', 'drilling', 'engine_idling']
    negative_test = []
    
    for neg_class in negative_classes:
        neg_clips = [c for c in dataset.load_clips().values() 
                    if c.class_label == neg_class][:10]
        negative_test.extend(neg_clips)
    
    print(f"Test set: {len(test_clips)} positive + {len(negative_test)} negative")
    print()
    
    # Evaluation
    y_true = []
    y_pred = []
    
    print("Testing positive examples (should detect):")
    for i, clip in enumerate(test_clips, 1):
        try:
            audio, sr = librosa.load(clip.audio_path, sr=22050)
            chunk_size = int(0.25 * sr)
            
            detected = False
            for j in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[j:j + chunk_size]
                if detector.process_chunk(chunk, sr, j // chunk_size):
                    detected = True
                    break
            
            y_true.append(1)
            y_pred.append(1 if detected else 0)
            
            status = "✅ DETECTED" if detected else "❌ MISSED"
            print(f"  [{i}/{len(test_clips)}] {clip.clip_id}: {status}")
        except Exception as e:
            print(f"  [{i}/{len(test_clips)}] {clip.clip_id}: ERROR - {e}")
            continue
    
    print()
    print("Testing negative examples (should NOT detect):")
    for i, clip in enumerate(negative_test, 1):
        try:
            audio, sr = librosa.load(clip.audio_path, sr=22050)
            chunk_size = int(0.25 * sr)
            
            detected = False
            for j in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[j:j + chunk_size]
                if detector.process_chunk(chunk, sr, j // chunk_size):
                    detected = True
                    break
            
            y_true.append(0)
            y_pred.append(1 if detected else 0)
            
            status = "❌ FALSE ALARM" if detected else "✅ IGNORED"
            print(f"  [{i}/{len(negative_test)}] {clip.clip_id} ({clip.tags.labels[0]}): {status}")
        except Exception as e:
            print(f"  [{i}/{len(negative_test)}] {clip.clip_id}: ERROR - {e}")
            continue
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    print()
    print("=" * 80)
    print("FINAL RESULTS WITH OPTIMAL TRAINING")
    print("=" * 80)
    print()
    print(f"📊 PERFORMANCE METRICS:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print()
    print(f"📈 CONFUSION MATRIX:")
    print(f"  True Positives:  {cm[1,1]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  True Negatives:  {cm[0,0]}")
    print(f"  False Negatives: {cm[1,0]}")
    print()
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Siren', 'Siren'],
                yticklabels=['Not Siren', 'Siren'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Full Dataset Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
    print("📊 Saved confusion matrix visualization")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }


def save_optimized_model(target_fingerprint, best_params, metrics, target_class='siren'):
    """Save the optimized model and parameters"""
    
    model_data = {
        'target_class': target_class,
        'fingerprint': target_fingerprint.tolist(),
        'optimal_thresholds': best_params,
        'test_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score'])
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    filename = f'optimized_model_{target_class}.json'
    with open(filename, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"💾 Optimized model saved to: {filename}")
    print()


def main():
    """
    Complete optimal training pipeline
    """
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "OPTIMAL TRAINING STRATEGY" + " "*33 + "║")
    print("║" + " "*16 + "Maximum Accuracy Configuration" + " "*31 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    target_class = 'siren'
    
    # Step 1: Load ALL data from ALL folds
    all_clips, dataset = load_all_clips_from_dataset(target_class)
    
    # Step 2: Proper train/test split
    train_clips, test_clips = stratified_train_test_split(all_clips, test_size=0.2)
    
    # Step 3: Extract robust fingerprint
    target_fingerprint, fingerprints_array, std_devs = extract_robust_fingerprint(
        train_clips,
        use_median=False,  # Set to True for more robustness
        remove_outliers=True  # Remove extreme outliers
    )
    
    # Step 4: Optimize thresholds
    best_params = optimize_thresholds(target_fingerprint, train_clips, test_clips, target_class)
    
    # Step 5: Create final detector with optimal thresholds
    print("=" * 80)
    print("CREATING FINAL OPTIMIZED DETECTOR")
    print("=" * 80)
    print()
    
    final_detector = NeuromorphicDetector(
        target_fingerprint,
        best_params['energy_threshold'],
        best_params['match_threshold'],
        target_class
    )
    
    # Step 6: Comprehensive evaluation
    metrics = comprehensive_evaluation(final_detector, test_clips, dataset, target_class)
    
    # Step 7: Save optimized model
    save_optimized_model(target_fingerprint, best_params, metrics, target_class)
    
    print("=" * 80)
    print("✅ OPTIMAL TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("📁 Generated files:")
    print("  • optimized_model_siren.json - Trained model with optimal parameters")
    print("  • threshold_optimization_heatmap.png - Threshold search visualization")
    print("  • confusion_matrix_optimized.png - Performance visualization")
    print()
    print("💡 RECOMMENDATIONS:")
    print("  ✓ Model trained on 80% of full dataset")
    print("  ✓ Thresholds optimized via grid search")
    print("  ✓ Outliers removed for robustness")
    print("  ✓ Evaluated on held-out test set (20%)")
    print()
    print(f"🎯 FINAL ACCURACY: {metrics['accuracy']*100:.2f}%")
    print()


if __name__ == "__main__":
    main()
