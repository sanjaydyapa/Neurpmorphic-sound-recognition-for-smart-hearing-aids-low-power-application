"""
GPU SNN TRAINING - FINAL PUSH TO 80%+
======================================
Optimized training strategy:
- Medium network: 512 → 384 → 256 → 10
- Lower initial learning rate: 0.0015
- More epochs: 200 with patience 30
- Batch normalization for stability
- No augmentation (original data quality is good)
- Adam optimizer with weight decay

Target: 80%+ accuracy
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("GPU SNN TRAINING - FINAL PUSH TO 80%+")
print("="*60)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
print("="*60)


class AudioSpikeEncoder:
    """Optimized feature extraction"""
    
    def __init__(self, n_mfcc=30, n_mels=128):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
    
    def extract_features(self, audio_path):
        """Extract optimized audio features"""
        try:
            y, sr = librosa.load(audio_path, duration=3.0, sr=22050)
            y = librosa.util.normalize(y)
            
            features = []
            
            # 1. MFCCs (30 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features.extend([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.max(mfcc, axis=1)
            ])
            
            # 2. Mel-Spectrogram (128 bands)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, 
                                                     n_fft=2048, hop_length=512)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features.extend([
                np.mean(mel_spec_db, axis=1),
                np.std(mel_spec_db, axis=1)
            ])
            
            # 3. Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth)
            ])
            
            # 4. Chroma
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1)
            ])
            
            # 5. ZCR and RMS
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            rms = librosa.feature.rms(y=y)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr),
                np.mean(rms),
                np.std(rms)
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
    """PyTorch dataset"""
    
    def __init__(self, features, labels, time_steps=10):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.time_steps = time_steps
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        spike_train = self.generate_spike_train(features)
        return spike_train, self.labels[idx]
    
    def generate_spike_train(self, features):
        """Generate Poisson spike train"""
        features_normalized = torch.sigmoid(features)
        spike_train = torch.rand(self.time_steps, len(features)) < features_normalized
        return spike_train.float()


class OptimizedSpikingAudioNet(nn.Module):
    """Optimized SNN with batch normalization"""
    
    def __init__(self, input_size, hidden_size1=512, hidden_size2=384, 
                 hidden_size3=256, num_classes=10, beta=0.95, dropout=0.25):
        super().__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_size3, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), output=True)
        
    def forward(self, x):
        """Forward pass"""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        mem4_rec = []
        
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            cur1 = self.bn1(cur1)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            
            cur2 = self.fc2(spk1)
            cur2 = self.bn2(cur2)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            
            cur3 = self.fc3(spk2)
            cur3 = self.bn3(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.dropout3(spk3)
            
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            mem4_rec.append(mem4)
        
        mem4_rec = torch.stack(mem4_rec)
        return mem4_rec[-1]


class FinalPushTrainer:
    """Optimized trainer for final push to 80%"""
    
    def __init__(self, data_path=r'C:\Users\sanjay\Documents\AIML-PROJECT\urbansound8k_data'):
        self.data_path = data_path
        self.encoder = AudioSpikeEncoder()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.device = device
        
    def load_and_extract_features(self):
        """Load dataset and extract features"""
        print("\nLoading dataset and extracting features...")
        
        csv_path = os.path.join(self.data_path, 'metadata', 'UrbanSound8K.csv')
        metadata = pd.read_csv(csv_path)
        
        features_list = []
        labels_list = []
        successful = 0
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting"):
            audio_path = os.path.join(
                self.data_path, 'audio',
                f"fold{row['fold']}", row['slice_file_name']
            )
            
            if not os.path.exists(audio_path):
                continue
            
            features = self.encoder.extract_features(audio_path)
            if features is not None and len(features) > 0:
                features_list.append(features)
                labels_list.append(row['classID'])
                successful += 1
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n✓ Samples: {X.shape[0]}")
        print(f"✓ Features: {X.shape[1]}")
        
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded, metadata['class'].unique()
    
    def train_model(self, X, y, class_names, epochs=200, batch_size=128, learning_rate=0.0015):
        """Train optimized model"""
        print(f"\n{'='*60}")
        print("TRAINING OPTIMIZED SNN - FINAL PUSH TO 80%+")
        print(f"{'='*60}")
        print(f"Samples: {X.shape[0]}")
        print(f"Features: {X.shape[1]}")
        print(f"Architecture: 512 → 384 → 256 → 10")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Patience: 30 epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        dataset = UrbanSoundDataset(X, y, time_steps=10)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=0, pin_memory=True)
        
        self.model = OptimizedSpikingAudioNet(
            input_size=X.shape[1],
            hidden_size1=512,
            hidden_size2=384,
            hidden_size3=256,
            num_classes=len(class_names),
            dropout=0.25
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, 
                                      weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )
        
        best_accuracy = 0
        patience = 30
        patience_counter = 0
        
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
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            scheduler.step(accuracy)
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                print(f"  ✓ NEW BEST: {best_accuracy:.2f}%")
                # Save best model
                torch.save(self.model.state_dict(), 'best_model_checkpoint.pth')
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n⚠️ Early stopping! Best: {best_accuracy:.2f}%")
                break
            
            print()
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model_checkpoint.pth'))
        
        # Final evaluation
        print("\nEvaluating best model...")
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
        """Save final model"""
        print("\nSaving final model...")
        
        save_dir = r'C:\Users\sanjay\Documents\AIML-PROJECT\trained_models'
        os.makedirs(save_dir, exist_ok=True)
        
        model_data = {
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'class_names': class_names,
            'accuracy': accuracy
        }
        
        model_path = os.path.join(save_dir, 'gpu_snn_final_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        metadata = {
            'accuracy': float(accuracy),
            'model_type': 'Optimized GPU SNN',
            'architecture': '512→384→256→10',
            'class_names': list(class_names)
        }
        
        metadata_path = os.path.join(save_dir, 'gpu_snn_final_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Model: {model_path}")
        print(f"✓ Metadata: {metadata_path}")
        print(f"✓ FINAL ACCURACY: {accuracy*100:.2f}%")
        print(f"{'='*60}")


def main():
    trainer = FinalPushTrainer()
    X, y, class_names = trainer.load_and_extract_features()
    accuracy = trainer.train_model(X, y, class_names, epochs=200, batch_size=128, learning_rate=0.0015)
    trainer.save_model(class_names, accuracy)
    
    print("\n" + "="*60)
    print("🎉 FINAL TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Accuracy: {accuracy*100:.2f}%")
    if accuracy >= 0.80:
        print("🎯 TARGET ACHIEVED: 80%+ accuracy!")
    print("="*60)


if __name__ == "__main__":
    main()
