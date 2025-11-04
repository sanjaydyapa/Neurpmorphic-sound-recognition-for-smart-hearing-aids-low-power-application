"""
Improved Multi-Class Sound Detector with Advanced Features
Uses ensemble methods and better feature engineering for higher accuracy
"""

import numpy as np
import librosa
import json
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

class ImprovedFeatureExtractor:
    """Extract rich audio features for better classification"""
    
    def __init__(self, sr=22050, n_mfcc=20, n_mels=128):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
    
    def extract_features(self, audio_path, duration=4.0):
        """Extract comprehensive audio features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
            
            # Ensure consistent length
            target_length = int(self.sr * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]
            
            features = []
            
            # 1. MFCCs (20 coefficients + delta + delta-delta)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            features.extend(mfcc_delta_mean)
            
            # 2. Mel-spectrogram features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_mean = np.mean(mel_spec_db, axis=1)
            mel_std = np.std(mel_spec_db, axis=1)
            
            features.extend(mel_mean)
            features.extend(mel_std)
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            features.extend(np.mean(spectral_contrast, axis=1))
            
            # 4. Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            
            # 5. Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # 6. RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            # 7. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            # Tempo might be an array, take first element or convert to scalar
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0] if len(tempo) > 0 else 120.0
            features.append(float(tempo))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None


class ImprovedSoundDetector:
    """Improved multi-class sound detector with ensemble learning"""
    
    def __init__(self, csv_path, audio_base_path):
        self.csv_path = csv_path
        self.audio_base_path = audio_base_path
        self.feature_extractor = ImprovedFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.sound_classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
    def load_dataset(self):
        """Load and prepare the dataset"""
        print("\n" + "="*60)
        print("📂 LOADING DATASET")
        print("="*60)
        
        # Load metadata
        metadata = []
        with open(self.csv_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                slice_file_name = parts[0]
                fold = int(parts[5])
                class_id = int(parts[6])
                class_name = parts[7]
                
                audio_path = os.path.join(self.audio_base_path, f"fold{fold}", slice_file_name)
                if os.path.exists(audio_path):
                    metadata.append({
                        'audio_path': audio_path,
                        'class_name': class_name,
                        'class_id': class_id,
                        'fold': fold
                    })
        
        print(f"✓ Loaded {len(metadata)} audio clips")
        
        # Get unique classes
        self.sound_classes = sorted(list(set([m['class_name'] for m in metadata])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.sound_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"✓ Found {len(self.sound_classes)} sound classes:")
        for cls in self.sound_classes:
            count = sum(1 for m in metadata if m['class_name'] == cls)
            print(f"   - {cls}: {count} clips")
        
        return metadata
    
    def extract_all_features(self, metadata):
        """Extract features from all audio files"""
        print("\n" + "="*60)
        print("🎵 EXTRACTING FEATURES")
        print("="*60)
        
        X = []
        y = []
        folds = []
        
        print("Extracting features from audio files...")
        for item in tqdm(metadata):
            features = self.feature_extractor.extract_features(item['audio_path'])
            if features is not None:
                X.append(features)
                y.append(self.class_to_idx[item['class_name']])
                folds.append(item['fold'])
        
        X = np.array(X)
        y = np.array(y)
        folds = np.array(folds)
        
        print(f"\n✓ Feature matrix shape: {X.shape}")
        print(f"✓ Total features per sample: {X.shape[1]}")
        
        return X, y, folds
    
    def train_with_cross_validation(self, X, y, folds):
        """Train model using cross-validation"""
        print("\n" + "="*60)
        print("🤖 TRAINING IMPROVED MODEL")
        print("="*60)
        
        # Check if we have enough data
        if len(X) < 100:
            print(f"⚠️ WARNING: Only {len(X)} valid samples extracted!")
            print("   Many audio files may be corrupted.")
            print("   Using 80-20 train-test split instead of fold-based.")
            
            # Use simple train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
        else:
            # Use fold-based cross-validation (like original UrbanSound8K)
            # Use folds 1-9 for training, fold 10 for final testing
            train_mask = folds != 10
            test_mask = folds == 10
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
        
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Test samples: {len(X_test)}")
        
        # Scale features
        print("\n📊 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try multiple models and pick the best
        print("\n🔍 Testing different models...")
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            # Removed Gradient Boosting - too slow for this dataset size
        }
        
        best_accuracy = 0
        best_model_name = None
        best_model = None
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"  Accuracy: {accuracy*100:.2f}%")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_model = model
        
        print(f"\n✓ Best Model: {best_model_name} ({best_accuracy*100:.2f}%)")
        self.model = best_model
        
        # Final evaluation
        print("\n" + "="*60)
        print("📊 FINAL EVALUATION")
        print("="*60)
        
        y_pred = self.model.predict(X_test_scaled)
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")
        
        # Per-class accuracy
        print("\n📈 Per-Class Accuracy:")
        class_accuracies = {}
        for idx, cls in self.idx_to_class.items():
            mask = y_test == idx
            if np.sum(mask) > 0:
                class_acc = accuracy_score(y_test[mask], y_pred[mask])
                class_accuracies[cls] = class_acc
                print(f"   {cls}: {class_acc*100:.2f}%")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.sound_classes))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'overall_accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test_scaled,
            'model_name': best_model_name
        }
    
    def save_model(self, results, output_dir='models'):
        """Save the trained model and results"""
        print("\n" + "="*60)
        print("💾 SAVING MODEL")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_dir, 'improved_sound_detector.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'sound_classes': self.sound_classes,
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class,
                'model_name': results['model_name']
            }, f)
        print(f"✓ Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': results['model_name'],
            'overall_accuracy': float(results['overall_accuracy']),
            'sound_classes': self.sound_classes,
            'class_accuracies': {k: float(v) for k, v in results['class_accuracies'].items()},
            'feature_count': self.feature_extractor.n_mfcc * 3 + self.feature_extractor.n_mels * 2 + 45
        }
        
        metadata_path = os.path.join(output_dir, 'improved_model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return model_path, metadata_path
    
    def plot_results(self, results, output_dir='images'):
        """Generate visualization plots"""
        print("\n" + "="*60)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=self.sound_classes,
                   yticklabels=self.sound_classes)
        plt.title('Improved Model - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'improved_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {cm_path}")
        plt.close()
        
        # 2. Per-Class Accuracy
        plt.figure(figsize=(14, 6))
        classes = list(results['class_accuracies'].keys())
        accuracies = [results['class_accuracies'][cls] * 100 for cls in classes]
        
        bars = plt.bar(range(len(classes)), accuracies, color='#3498db', alpha=0.8)
        plt.axhline(y=results['overall_accuracy']*100, color='red', linestyle='--', 
                   label=f'Overall: {results["overall_accuracy"]*100:.1f}%', linewidth=2)
        
        plt.xlabel('Sound Class', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Improved Model - Per-Class Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        acc_path = os.path.join(output_dir, 'improved_per_class_accuracy.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class accuracy plot saved to: {acc_path}")
        plt.close()
        
        # 3. Comparison with baseline
        plt.figure(figsize=(10, 6))
        comparison = {
            'Random\nGuessing': 10,
            'Old Model\n(MFCC only)': 19.98,
            'Improved Model\n(Rich Features + ML)': results['overall_accuracy'] * 100
        }
        
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        bars = plt.bar(comparison.keys(), comparison.values(), color=colors, alpha=0.8)
        
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Model Comparison: Accuracy Improvement', fontsize=16, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, (name, val) in zip(bars, comparison.items()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        comp_path = os.path.join(output_dir, 'model_comparison.png')
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {comp_path}")
        plt.close()


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🚀 IMPROVED MULTI-CLASS SOUND DETECTOR TRAINING")
    print("="*70)
    
    # Paths
    csv_path = "../urbansound8k_data/metadata/UrbanSound8K.csv"
    audio_base_path = "../urbansound8k_data/audio"
    
    # Initialize detector
    detector = ImprovedSoundDetector(csv_path, audio_base_path)
    
    # Load dataset
    metadata = detector.load_dataset()
    
    # Extract features
    X, y, folds = detector.extract_all_features(metadata)
    
    # Train model
    results = detector.train_with_cross_validation(X, y, folds)
    
    # Save model
    detector.save_model(results)
    
    # Generate visualizations
    detector.plot_results(results)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n🎯 Final Accuracy: {results['overall_accuracy']*100:.2f}%")
    print(f"🔥 Improvement: {results['overall_accuracy']*100 - 19.98:.2f}% over baseline")
    print("\nNext steps:")
    print("1. Check the visualizations in the 'images' folder")
    print("2. Update the web demo to use the improved model")
    print("3. Test the improved model on real audio samples")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
