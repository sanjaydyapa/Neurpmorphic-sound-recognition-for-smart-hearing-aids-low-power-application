"""
Fast XGBoost Training - Simplified Feature Set for 80%+ Accuracy
Uses proven features that work well, trains quickly
"""

import numpy as np
import librosa
import json
import os
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')


class FastFeatureExtractor:
    """Extract proven effective features quickly"""
    
    def __init__(self, sr=22050):
        self.sr = sr
        
    def extract_features(self, audio_path, duration=4.0):
        """Extract ~500 effective features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
            
            # Ensure consistent length
            target_length = int(sr * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]
            
            features = []
            
            # 1. MFCCs (30 coefficients with statistics)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            features.extend(np.max(mfcc, axis=1))
            features.extend(np.min(mfcc, axis=1))
            
            # MFCC deltas
            mfcc_delta = librosa.feature.delta(mfcc)
            features.extend(np.mean(mfcc_delta, axis=1))
            features.extend(np.std(mfcc_delta, axis=1))
            
            # 2. Mel-Spectrogram (128 bands)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features.extend(np.mean(mel_spec_db, axis=1))
            features.extend(np.std(mel_spec_db, axis=1))
            
            # 3. Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            
            for feat in [spectral_centroids, spectral_rolloff, spectral_flatness]:
                features.extend([np.mean(feat), np.std(feat), np.max(feat), np.min(feat)])
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
            features.extend(np.mean(spectral_contrast, axis=1))
            features.extend(np.std(spectral_contrast, axis=1))
            
            # 4. Chroma Features
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma_stft, axis=1))
            features.extend(np.std(chroma_stft, axis=1))
            
            # 5. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)])
            
            # 6. RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])
            
            # 7. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            # 8. Onset Strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features.extend([np.mean(onset_env), np.std(onset_env), np.max(onset_env)])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error: {audio_path}: {str(e)[:50]}")
            return None


class FastXGBoostDetector:
    """Fast XGBoost detector"""
    
    def __init__(self, csv_path, audio_base_path):
        self.csv_path = csv_path
        self.audio_base_path = audio_base_path
        self.feature_extractor = FastFeatureExtractor()
        
    def load_dataset(self):
        """Load dataset from CSV"""
        import pandas as pd
        
        print("\n" + "="*60)
        print("📂 LOADING DATASET")
        print("="*60)
        
        df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(df)} audio clips")
        
        metadata = []
        for _, row in df.iterrows():
            fold = f"fold{row['fold']}"
            filename = row['slice_file_name']
            class_name = row['class']
            audio_path = os.path.join(self.audio_base_path, fold, filename)
            
            metadata.append({
                'audio_path': audio_path,
                'class': class_name,
                'fold': row['fold']
            })
        
        classes = df['class'].value_counts()
        print(f"✓ Found {len(classes)} sound classes:")
        for cls, count in classes.items():
            print(f"   - {cls}: {count} clips")
        
        return metadata
    
    def extract_all_features(self, metadata):
        """Extract features from all audio files"""
        print("\n" + "="*60)
        print("🎵 EXTRACTING FEATURES (~500 per clip)")
        print("="*60)
        print("Estimated time: 10-15 minutes...")
        
        features_list = []
        labels = []
        folds = []
        
        for item in tqdm(metadata, desc="Processing"):
            features = self.feature_extractor.extract_features(item['audio_path'])
            if features is not None:
                features_list.append(features)
                labels.append(item['class'])
                folds.append(item['fold'])
        
        X = np.array(features_list)
        y = np.array(labels)
        folds = np.array(folds)
        
        print(f"\n✓ Extracted features from {len(X)} clips")
        print(f"✓ Feature dimension: {X.shape[1]} features per clip")
        
        return X, y, folds
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Optimized XGBoost
        model = XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        model.fit(X_train_scaled, y_train_encoded, verbose=False)
        
        y_pred_encoded = model.predict(X_test_scaled)
        y_pred = le.inverse_transform(y_pred_encoded)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, scaler, le, accuracy, y_pred
    
    def train_and_evaluate(self, X, y, folds):
        """Train with 10-fold cross-validation"""
        print("\n" + "="*60)
        print("🚀 TRAINING XGBOOST MODEL")
        print("="*60)
        
        fold_accuracies = []
        all_predictions = []
        all_true_labels = []
        
        unique_folds = np.unique(folds)
        
        for test_fold in tqdm(unique_folds, desc="Training folds"):
            train_mask = folds != test_fold
            test_mask = folds == test_fold
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            model, scaler, le, accuracy, y_pred = self.train_xgboost(
                X_train, y_train, X_test, y_test
            )
            
            fold_accuracies.append(accuracy)
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_test)
            
            print(f"  Fold {test_fold}: {accuracy*100:.2f}%")
        
        overall_accuracy = np.mean(fold_accuracies)
        
        print(f"\n{'='*60}")
        print(f"🎯 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {overall_accuracy*100:.2f}%")
        
        # Retrain on all data
        print("\n📦 Training final model on all data...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        final_model = XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        final_model.fit(X_scaled, y_encoded)
        
        # Per-class accuracies
        class_accuracies = {}
        unique_classes = np.unique(all_true_labels)
        print("\nPer-Class Accuracy:")
        for cls in unique_classes:
            mask = np.array(all_true_labels) == cls
            cls_acc = accuracy_score(
                np.array(all_true_labels)[mask],
                np.array(all_predictions)[mask]
            )
            class_accuracies[cls] = cls_acc
            print(f"  {cls}: {cls_acc*100:.2f}%")
        
        return final_model, scaler, le, overall_accuracy, class_accuracies, all_true_labels, all_predictions
    
    def save_model(self, model, scaler, le, accuracy, class_accuracies, output_dir='models'):
        """Save the trained model"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'label_encoder': le,
            'sound_classes': list(le.classes_),
            'overall_accuracy': accuracy,
            'class_accuracies': class_accuracies
        }
        
        model_path = os.path.join(output_dir, 'xgboost_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': 'XGBoost Fast',
            'overall_accuracy': float(accuracy),
            'sound_classes': list(le.classes_),
            'class_accuracies': {k: float(v) for k, v in class_accuracies.items()},
            'feature_count': model.n_features_in_
        }
        
        metadata_path = os.path.join(output_dir, 'xgboost_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✓ Metadata saved to {metadata_path}")
    
    def visualize_results(self, y_true, y_pred, class_accuracies, overall_accuracy, output_dir='images'):
        """Create visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(14, 12))
        cm = confusion_matrix(y_true, y_pred)
        classes = sorted(set(y_true))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes,
                   cbar_kws={'label': 'Count'})
        plt.title(f'XGBoost Model Confusion Matrix\nOverall Accuracy: {overall_accuracy*100:.2f}%',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('True Class', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgboost_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix saved")
        plt.close()
        
        # Per-class accuracy
        plt.figure(figsize=(14, 8))
        classes_sorted = sorted(class_accuracies.keys())
        accuracies = [class_accuracies[cls]*100 for cls in classes_sorted]
        colors = ['#27ae60' if acc >= 80 else '#f39c12' if acc >= 70 else '#e74c3c' for acc in accuracies]
        
        bars = plt.bar(range(len(classes_sorted)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        plt.xlabel('Sound Class', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        plt.title('XGBoost: Per-Class Performance', fontsize=16, fontweight='bold')
        plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.axhline(y=overall_accuracy*100, color='blue', linestyle='--', linewidth=2, label=f'Overall: {overall_accuracy*100:.1f}%')
        plt.grid(axis='y', alpha=0.3)
        plt.legend(fontsize=12)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgboost_per_class_accuracy.png'), dpi=300, bbox_inches='tight')
        print("✓ Per-class accuracy chart saved")
        plt.close()
        
        # Model comparison
        plt.figure(figsize=(12, 8))
        comparison = {
            'Baseline\n(MFCC)': 19.98,
            'Random Forest\n(361 features)': 72.52,
            'XGBoost\n(~500 features)': overall_accuracy * 100
        }
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        bars = plt.bar(comparison.keys(), comparison.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        plt.title('Model Evolution & Improvement', fontsize=18, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, (name, val) in zip(bars, comparison.items()):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'xgboost_model_comparison.png'), dpi=300, bbox_inches='tight')
        print("✓ Model comparison chart saved")
        plt.close()


def main():
    print("\n" + "="*70)
    print("🚀 FAST XGBOOST TRAINING")
    print("   Target: 80%+ Accuracy with Efficient Feature Set")
    print("="*70)
    
    csv_path = "../urbansound8k_data/metadata/UrbanSound8K.csv"
    audio_base_path = "../urbansound8k_data/audio"
    
    detector = FastXGBoostDetector(csv_path, audio_base_path)
    
    # Load dataset
    metadata = detector.load_dataset()
    
    # Extract features
    X, y, folds = detector.extract_all_features(metadata)
    
    # Train and evaluate
    model, scaler, le, accuracy, class_acc, y_true, y_pred = detector.train_and_evaluate(X, y, folds)
    
    # Save model
    detector.save_model(model, scaler, le, accuracy, class_acc)
    
    # Visualize
    detector.visualize_results(y_true, y_pred, class_acc, accuracy)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print(f"🎯 Final Accuracy: {accuracy*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
