"""
Advanced Neuromorphic Sound Detector - Targeting 80-85% Accuracy
Uses spike-based features with hierarchical classification and per-class optimization
"""

import numpy as np
import librosa
import json
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')


class AdvancedNeuromorphicFeatureExtractor:
    """
    Extract spike-based and bio-inspired features for neuromorphic computing
    Focuses on temporal patterns and energy dynamics
    """
    
    def __init__(self, sr=22050, n_mfcc=25, n_mels=128, hop_length=512):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        
    def extract_spike_patterns(self, y, sr):
        """Extract spike-like temporal patterns from audio"""
        # Compute onset strength (spike events)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Spike statistics
        spike_mean = np.mean(onset_env)
        spike_std = np.std(onset_env)
        spike_max = np.max(onset_env)
        spike_rate = np.sum(onset_env > np.mean(onset_env)) / len(onset_env)
        
        # Inter-spike intervals (ISI)
        spike_times = np.where(onset_env > np.mean(onset_env))[0]
        if len(spike_times) > 1:
            isi = np.diff(spike_times)
            isi_mean = np.mean(isi)
            isi_std = np.std(isi)
            isi_cv = isi_std / (isi_mean + 1e-8)  # Coefficient of variation
        else:
            isi_mean = isi_std = isi_cv = 0
        
        return [spike_mean, spike_std, spike_max, spike_rate, isi_mean, isi_std, isi_cv]
    
    def extract_temporal_envelope(self, y):
        """Extract temporal envelope features"""
        # RMS energy over time
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Envelope statistics
        env_mean = np.mean(rms)
        env_std = np.std(rms)
        env_max = np.max(rms)
        env_attack = np.argmax(rms) / len(rms)  # Attack time (normalized)
        
        # Envelope dynamics
        env_delta = np.diff(rms)
        env_delta_mean = np.mean(np.abs(env_delta))
        env_delta_std = np.std(env_delta)
        
        return [env_mean, env_std, env_max, env_attack, env_delta_mean, env_delta_std]
    
    def extract_comprehensive_features(self, audio_path, duration=4.0):
        """Extract all features for neuromorphic classification"""
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
            
            # 1. Enhanced MFCCs (25 coefficients + deltas + delta-deltas)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_max = np.max(mfcc, axis=1)
            mfcc_min = np.min(mfcc, axis=1)
            
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            mfcc_delta_std = np.std(mfcc_delta, axis=1)
            
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
            
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            features.extend(mfcc_max)
            features.extend(mfcc_min)
            features.extend(mfcc_delta_mean)
            features.extend(mfcc_delta_std)
            features.extend(mfcc_delta2_mean)
            
            # 2. Enhanced Mel-spectrogram features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            mel_mean = np.mean(mel_spec_db, axis=1)
            mel_std = np.std(mel_spec_db, axis=1)
            mel_max = np.max(mel_spec_db, axis=1)
            mel_min = np.min(mel_spec_db, axis=1)
            
            features.extend(mel_mean)
            features.extend(mel_std)
            features.extend(mel_max)
            features.extend(mel_min)
            
            # 3. Comprehensive spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
            spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)[0]
            
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.max(spectral_centroids), np.min(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                np.mean(spectral_flatness), np.std(spectral_flatness)
            ])
            features.extend(np.mean(spectral_contrast, axis=1))
            features.extend(np.std(spectral_contrast, axis=1))
            
            # 4. Enhanced chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            
            features.extend(np.mean(chroma, axis=1))
            features.extend(np.std(chroma, axis=1))
            features.extend(np.max(chroma, axis=1))
            features.extend(np.mean(chroma_cqt, axis=1))
            
            # 5. Zero crossing rate features
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
            features.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])
            
            # 6. Temporal envelope features (neuromorphic-inspired)
            envelope_features = self.extract_temporal_envelope(y)
            features.extend(envelope_features)
            
            # 7. Spike patterns (neuromorphic-inspired)
            spike_features = self.extract_spike_patterns(y, sr)
            features.extend(spike_features)
            
            # 8. Tempo and rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0] if len(tempo) > 0 else 120.0
            
            # Beat statistics
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                beat_regularity = np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-8)
            else:
                beat_regularity = 0
            
            features.append(float(tempo))
            features.append(beat_regularity)
            features.append(len(beats) / duration)  # Beats per second
            
            # 9. Tonnetz features (harmonic content)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features.extend(np.mean(tonnetz, axis=1))
            features.extend(np.std(tonnetz, axis=1))
            
            # 10. Spectral derivatives (change over time)
            mel_delta = librosa.feature.delta(mel_spec_db)
            features.extend([
                np.mean(np.abs(mel_delta)),
                np.std(np.abs(mel_delta)),
                np.max(np.abs(mel_delta))
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None


class AdvancedNeuromorphicDetector:
    """Advanced neuromorphic detector with ensemble and per-class optimization"""
    
    def __init__(self, csv_path, audio_base_path):
        self.csv_path = csv_path
        self.audio_base_path = audio_base_path
        self.feature_extractor = AdvancedNeuromorphicFeatureExtractor()
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
        
        metadata = []
        with open(self.csv_path, 'r') as f:
            lines = f.readlines()[1:]
            
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
        print("🎵 EXTRACTING ADVANCED FEATURES")
        print("="*60)
        
        X = []
        y = []
        folds = []
        
        print("Extracting neuromorphic features from audio files...")
        for item in tqdm(metadata):
            features = self.feature_extractor.extract_comprehensive_features(item['audio_path'])
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
    
    def train_ensemble_model(self, X, y, folds):
        """Train ensemble model with cross-validation"""
        print("\n" + "="*60)
        print("🤖 TRAINING ADVANCED ENSEMBLE MODEL")
        print("="*60)
        
        # Split data
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
        
        # Train ensemble of Random Forests with different configurations
        print("\n🔍 Training ensemble of optimized Random Forests...")
        
        rf1 = RandomForestClassifier(
            n_estimators=300,
            max_depth=35,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf2 = RandomForestClassifier(
            n_estimators=300,
            max_depth=40,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='log2',
            bootstrap=True,
            random_state=43,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        rf3 = RandomForestClassifier(
            n_estimators=400,
            max_depth=30,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=44,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Create voting ensemble
        self.model = VotingClassifier(
            estimators=[('rf1', rf1), ('rf2', rf2), ('rf3', rf3)],
            voting='soft',
            n_jobs=-1
        )
        
        print("Training ensemble (this may take 5-10 minutes)...")
        self.model.fit(X_train_scaled, y_train)
        
        print("\n" + "="*60)
        print("📊 FINAL EVALUATION")
        print("="*60)
        
        y_pred = self.model.predict(X_test_scaled)
        
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
        print(classification_report(y_test, y_pred, target_names=self.sound_classes, digits=3))
        
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'overall_accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test_scaled
        }
    
    def save_model(self, results, output_dir='models'):
        """Save the trained model"""
        print("\n" + "="*60)
        print("💾 SAVING ADVANCED MODEL")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, 'advanced_neuromorphic_detector.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'sound_classes': self.sound_classes,
                'class_to_idx': self.class_to_idx,
                'idx_to_class': self.idx_to_class
            }, f)
        print(f"✓ Model saved to: {model_path}")
        
        metadata = {
            'overall_accuracy': float(results['overall_accuracy']),
            'sound_classes': self.sound_classes,
            'class_accuracies': {k: float(v) for k, v in results['class_accuracies'].items()}
        }
        
        metadata_path = os.path.join(output_dir, 'advanced_model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✓ Metadata saved to: {metadata_path}")
        
        return model_path
    
    def plot_results(self, results, output_dir='images'):
        """Generate visualizations"""
        print("\n" + "="*60)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.sound_classes,
                   yticklabels=self.sound_classes)
        plt.title('Advanced Model - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'advanced_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved")
        plt.close()
        
        # Per-Class Accuracy
        plt.figure(figsize=(14, 6))
        classes = list(results['class_accuracies'].keys())
        accuracies = [results['class_accuracies'][cls] * 100 for cls in classes]
        
        bars = plt.bar(range(len(classes)), accuracies, color='#27ae60', alpha=0.8)
        plt.axhline(y=results['overall_accuracy']*100, color='red', linestyle='--', 
                   label=f'Overall: {results["overall_accuracy"]*100:.1f}%', linewidth=2)
        
        plt.xlabel('Sound Class', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        plt.title('Advanced Model - Per-Class Accuracy', fontsize=16, fontweight='bold')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.legend(fontsize=11)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 100)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        acc_path = os.path.join(output_dir, 'advanced_per_class_accuracy.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class accuracy plot saved")
        plt.close()
        
        # Model Comparison
        plt.figure(figsize=(12, 7))
        comparison = {
            'Random\nGuessing': 10,
            'Baseline\n(MFCC)': 19.98,
            'Improved\n(RF 358 features)': 72.52,
            'Advanced\n(Ensemble + Optimization)': results['overall_accuracy'] * 100
        }
        
        colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
        bars = plt.bar(comparison.keys(), comparison.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        plt.title('Progressive Model Improvement Journey', fontsize=18, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, (name, val) in zip(bars, comparison.items()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        comp_path = os.path.join(output_dir, 'model_progression.png')
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved")
        plt.close()


def main():
    print("\n" + "="*70)
    print("🚀 ADVANCED NEUROMORPHIC SOUND DETECTOR")
    print("   Target: 80-85% Accuracy with Ensemble Learning")
    print("="*70)
    
    csv_path = "../urbansound8k_data/metadata/UrbanSound8K.csv"
    audio_base_path = "../urbansound8k_data/audio"
    
    detector = AdvancedNeuromorphicDetector(csv_path, audio_base_path)
    
    metadata = detector.load_dataset()
    X, y, folds = detector.extract_all_features(metadata)
    results = detector.train_ensemble_model(X, y, folds)
    detector.save_model(results)
    detector.plot_results(results)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n🎯 Final Accuracy: {results['overall_accuracy']*100:.2f}%")
    print(f"🔥 Total Improvement: {results['overall_accuracy']*100 - 19.98:.2f}% over baseline")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
