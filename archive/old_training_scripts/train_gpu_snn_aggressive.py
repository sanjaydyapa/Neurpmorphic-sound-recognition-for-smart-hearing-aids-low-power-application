"""
GPU-ACCELERATED SNN TRAINING - AGGRESSIVE VERSION
==================================================
Enhanced training with:
- Larger network (768 → 512 → 256 neurons)
- Data augmentation (noise, pitch, time stretch)
- Aggressive training (150 epochs)
- Dropout regularization
- Better learning rate scheduling

Target: 85%+ accuracy
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("AGGRESSIVE GPU-ACCELERATED SNN TRAINING")
print("="*60)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*60)


class AudioSpikeEncoder:
    """Enhanced feature extraction with data augmentation"""
    
    def __init__(self, n_mfcc=40, n_mels=128):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
    
    def augment_audio(self, y, sr, augment=True):
        """Apply random augmentation to audio"""
        if not augment:
            return y
        
        # Random noise
        if np.random.random() < 0.3:
            noise = np.random.randn(len(y)) * 0.005
            y = y + noise
        
        # Random pitch shift
        if np.random.random() < 0.3:
            n_steps = np.random.randint(-2, 3)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        
        # Random time stretch
        if np.random.random() < 0.3:
            rate = np.random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
        
        # Pad or trim to 3 seconds
        target_length = sr * 3
        if len(y) > target_length:
            y = y[:target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        
        return y
    
    def extract_features(self, audio_path, augment=False):
        """Extract comprehensive audio features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, duration=3.0, sr=22050)
            y = librosa.util.normalize(y)
            
            # Apply augmentation if enabled
            if augment:
                y = self.augment_audio(y, sr)
            
            features = []
            
            # 1. MFCCs (40 coefficients with more stats)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features.extend([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.max(mfcc, axis=1),
                np.min(mfcc, axis=1)
            ])
            
            # 2. Mel-Spectrogram (128 bands)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features.extend([
                np.mean(mel_spec_db, axis=1),
                np.std(mel_spec_db, axis=1),
                np.max(mel_spec_db, axis=1)
            ])
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.max(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.max(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.mean(spectral_contrast, axis=1),
                np.std(spectral_contrast, axis=1)
            ])
            
            # 4. Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1),
                np.max(chroma, axis=1)
            ])
            
            # 5. Tonnetz
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features.extend([
                np.mean(tonnetz, axis=1),
                np.std(tonnetz, axis=1)
            ])
            
            # 6. ZCR and RMS
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            rms = librosa.feature.rms(y=y)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr),
                np.max(zcr),
                np.mean(rms),
                np.std(rms),
                np.max(rms)
            ])
            
            # Flatten
            feature_vector = []
            for feat in features:
                if isinstance(feat, np.ndarray):
                    feature_vector.extend(feat)
                else:
                    feature_vector.append(feat)
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            return None


class UrbanSoundDataset(Dataset):
    """Enhanced PyTorch dataset with augmentation support"""
    
    def __init__(self, features, labels, time_steps=10):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.time_steps = time_steps
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Convert features to spike trains
        features = self.features[idx]
        spike_train = self.generate_spike_train(features)
        return spike_train, self.labels[idx]
    
    def generate_spike_train(self, features):
        """Generate Poisson spike train from features"""
        # Normalize to [0, 1]
        features_normalized = torch.sigmoid(features)
        
        # Generate spike train
        spike_train = torch.rand(self.time_steps, len(features)) < features_normalized
        return spike_train.float()


class EnhancedSpikingAudioNet(nn.Module):
    """Enhanced Spiking Neural Network with larger capacity"""
    
    def __init__(self, input_size, hidden_size1=768, hidden_size2=512, hidden_size3=256, 
                 num_classes=10, beta=0.95, dropout=0.3):
        super().__init__()
        
        # Larger network architecture
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(hidden_size3, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), output=True)
        
    def forward(self, x):
        """Forward pass through SNN"""
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        # Process spike train over time
        spk4_rec = []
        mem4_rec = []
        
        for step in range(x.size(0)):  # Iterate over time steps
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.dropout3(spk3)
            
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            spk4_rec.append(spk4)
            mem4_rec.append(mem4)
        
        spk4_rec = torch.stack(spk4_rec)
        mem4_rec = torch.stack(mem4_rec)
        
        return mem4_rec[-1]


class AggressiveGPUSNNTrainer:
    """Aggressive trainer with enhanced features"""
    
    def __init__(self, data_path=r'C:\Users\sanjay\Documents\AIML-PROJECT\urbansound8k_data'):
        self.data_path = data_path
        self.encoder = AudioSpikeEncoder(n_mfcc=40, n_mels=128)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = device
        
    def load_and_extract_features(self, augment_multiplier=2):
        """Load dataset and extract features with augmentation"""
        print("\nLoading dataset and extracting features...")
        print(f"Augmentation: {augment_multiplier}x data")
        
        csv_path = os.path.join(self.data_path, 'metadata', 'UrbanSound8K.csv')
        metadata = pd.read_csv(csv_path)
        
        features_list = []
        labels_list = []
        successful = 0
        errors = 0
        
        # Original data
        print("\n1. Extracting original features...")
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Original"):
            audio_path = os.path.join(
                self.data_path, 'audio',
                f"fold{row['fold']}", row['slice_file_name']
            )
            
            if not os.path.exists(audio_path):
                errors += 1
                continue
            
            features = self.encoder.extract_features(audio_path, augment=False)
            if features is not None and len(features) > 0:
                features_list.append(features)
                labels_list.append(row['classID'])
                successful += 1
            else:
                errors += 1
        
        # Augmented data
        if augment_multiplier > 1:
            print(f"\n2. Generating {augment_multiplier-1}x augmented data...")
            for aug_round in range(augment_multiplier - 1):
                for idx, row in tqdm(metadata.iterrows(), total=len(metadata), 
                                   desc=f"Augmentation {aug_round+1}"):
                    audio_path = os.path.join(
                        self.data_path, 'audio',
                        f"fold{row['fold']}", row['slice_file_name']
                    )
                    
                    if not os.path.exists(audio_path):
                        continue
                    
                    features = self.encoder.extract_features(audio_path, augment=True)
                    if features is not None and len(features) > 0:
                        features_list.append(features)
                        labels_list.append(row['classID'])
                        successful += 1
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n✓ Total samples: {X.shape[0]}")
        print(f"✓ Features per sample: {X.shape[1]}")
        print(f"✓ Success rate: {successful/(successful+errors)*100:.1f}%")
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded, metadata['class'].unique()
    
    def train_model(self, X, y, class_names, epochs=150, batch_size=256, learning_rate=0.002):
        """Train enhanced SNN model"""
        print(f"\n{'='*60}")
        print("TRAINING ENHANCED GPU SNN")
        print(f"{'='*60}")
        print(f"Samples: {X.shape[0]}")
        print(f"Features: {X.shape[1]}")
        print(f"Classes: {len(class_names)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Initial LR: {learning_rate}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Dataset
        dataset = UrbanSoundDataset(X, y, time_steps=10)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=0, pin_memory=True)
        
        # Enhanced model
        self.model = EnhancedSpikingAudioNet(
            input_size=X.shape[1],
            hidden_size1=768,
            hidden_size2=512,
            hidden_size3=256,
            num_classes=len(class_names),
            dropout=0.3
        ).to(self.device)
        
        # Optimizer with weight decay
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, 
                                      weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Early stopping
        best_accuracy = 0
        patience = 20
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for spike_trains, labels in pbar:
                spike_trains = spike_trains.to(self.device)
                labels = labels.to(self.device)
                spike_trains = spike_trains.permute(1, 0, 2)
                
                outputs = self.model(spike_trains)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            
            scheduler.step()
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                print(f"  ✓ New best accuracy!")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping! Best: {best_accuracy:.2f}%")
                break
            
            print()
        
        # Final evaluation
        print("\nEvaluating final model...")
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for spike_trains, labels in tqdm(train_loader, desc="Evaluation"):
                spike_trains = spike_trains.to(self.device)
                spike_trains = spike_trains.permute(1, 0, 2)
                outputs = self.model(spike_trains)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        final_accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        print("\nPer-class accuracy:")
        for i, class_name in enumerate(class_names):
            class_mask = np.array(all_labels) == i
            class_acc = accuracy_score(
                np.array(all_labels)[class_mask],
                np.array(all_preds)[class_mask]
            )
            print(f"  {class_name:20s}: {class_acc*100:6.2f}%")
        
        return final_accuracy
    
    def save_model(self, class_names, accuracy):
        """Save model"""
        print("\nSaving enhanced SNN model...")
        
        save_dir = r'C:\Users\sanjay\Documents\AIML-PROJECT\trained_models'
        os.makedirs(save_dir, exist_ok=True)
        
        model_data = {
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'class_names': class_names,
            'accuracy': accuracy
        }
        
        model_path = os.path.join(save_dir, 'gpu_snn_aggressive_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        metadata = {
            'accuracy': float(accuracy),
            'model_type': 'Enhanced GPU-Accelerated SNN',
            'architecture': '4-layer (768→512→256→10)',
            'training': 'Aggressive with augmentation',
            'class_names': list(class_names)
        }
        
        metadata_path = os.path.join(save_dir, 'gpu_snn_aggressive_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Model: {model_path}")
        print(f"✓ Metadata: {metadata_path}")
        print(f"✓ Accuracy: {accuracy*100:.2f}%")
        print(f"✓ Enhanced SNN with augmentation")
        print(f"{'='*60}")


def main():
    print("\n" + "="*60)
    print("AGGRESSIVE GPU SNN TRAINING")
    print("="*60)
    print("\nEnhancements:")
    print("  • Larger network: 768 → 512 → 256 → 10")
    print("  • 2x data augmentation")
    print("  • 150 epochs with cosine annealing")
    print("  • Dropout regularization")
    print("  • Gradient clipping")
    print("\nTarget: 85%+ accuracy\n")
    
    trainer = AggressiveGPUSNNTrainer()
    
    # Extract features with 2x augmentation
    X, y, class_names = trainer.load_and_extract_features(augment_multiplier=2)
    
    # Train aggressively
    accuracy = trainer.train_model(X, y, class_names, epochs=150, batch_size=256)
    
    # Save
    trainer.save_model(class_names, accuracy)
    
    print("\n" + "="*60)
    print("AGGRESSIVE TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")
    print("\nModel ready for deployment!")
    print("="*60)


if __name__ == "__main__":
    main()
