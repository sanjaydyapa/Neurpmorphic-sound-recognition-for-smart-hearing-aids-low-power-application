"""
Flask Backend Server for Improved Sound Detector Demo
Uses the improved Random Forest model with 72.52% accuracy
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import random
import os
import time
import numpy as np
import librosa
import pickle

app = Flask(__name__)
CORS(app)

# Global variables
MODEL_DATA = None
AUDIO_BASE_PATH = "../urbansound8k_data/audio"
CSV_PATH = "../urbansound8k_data/metadata/UrbanSound8K.csv"

class ImprovedDetector:
    """Improved sound detector using Random Forest with rich features"""
    
    def __init__(self, model_path):
        print(f"Loading improved model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.sound_classes = model_data['sound_classes']
        self.class_to_idx = model_data['class_to_idx']
        self.idx_to_class = model_data['idx_to_class']
        print("✓ Improved model loaded successfully!")
    
    def extract_features(self, audio_path, sr=22050, duration=4.0):
        """Extract the same 358 features used in training"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
            
            # Ensure consistent length
            target_length = int(sr * duration)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                y = y[:target_length]
            
            features = []
            
            # 1. MFCCs (20 coefficients + delta + delta-delta)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
            
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            features.extend(mfcc_delta_mean)
            
            # 2. Mel-spectrogram features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
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
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0] if len(tempo) > 0 else 120.0
            features.append(float(tempo))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def classify(self, audio_path):
        """Classify audio file using improved model"""
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(audio_path)
        
        if features is None:
            return {
                'predicted_class': 'error',
                'confidence': 0,
                'processing_time': time.time() - start_time
            }
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        predicted_class = self.idx_to_class[prediction]
        confidence = probabilities[prediction] * 100
        
        processing_time = time.time() - start_time
        
        return {
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'processing_time': round(processing_time, 3),
            'all_probabilities': {
                self.idx_to_class[i]: round(prob * 100, 2) 
                for i, prob in enumerate(probabilities)
            }
        }


def load_model():
    """Load the demo-ready ensemble model"""
    global MODEL_DATA
    try:
        model_path = '../trained_models/demo_ready_model.pkl'
        MODEL_DATA = ImprovedDetector(model_path)
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def load_audio_metadata():
    """Load audio metadata from CSV"""
    metadata = {}
    try:
        with open(CSV_PATH, 'r') as f:
            lines = f.readlines()[1:]
            
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    slice_file_name = parts[0]
                    fold = int(parts[5])
                    class_id = int(parts[6])
                    class_name = parts[7]
                    
                    metadata[slice_file_name] = {
                        'fold': fold,
                        'class_id': class_id,
                        'class_name': class_name
                    }
        
        print(f"✓ Loaded metadata for {len(metadata)} audio clips")
        return metadata
    except Exception as e:
        print(f"✗ Error loading metadata: {e}")
        return {}


def get_random_audio_clip(class_filter=None, fold_filter=None, metadata=None):
    """Get a random audio clip based on filters"""
    if metadata is None:
        metadata = load_audio_metadata()
    
    valid_clips = []
    for clip_name, info in metadata.items():
        if class_filter and class_filter != 'random':
            if info['class_name'] != class_filter:
                continue
        
        if fold_filter and fold_filter != 'random':
            if info['fold'] != int(fold_filter):
                continue
        
        audio_path = os.path.join(AUDIO_BASE_PATH, f"fold{info['fold']}", clip_name)
        if os.path.exists(audio_path):
            valid_clips.append((clip_name, info, audio_path))
    
    if not valid_clips:
        return None, None, None
    
    clip_name, info, audio_path = random.choice(valid_clips)
    return clip_name, info, audio_path


@app.route('/')
def index():
    return send_from_directory('.', 'demo.html')


@app.route('/demo_details.html')
def details():
    return send_from_directory('.', 'demo_details.html')


@app.route('/<path:filename>')
def serve_static(filename):
    try:
        if filename.endswith('.json'):
            return send_from_directory('../models', filename)
        if filename.startswith('visualizations/'):
            return send_from_directory('..', filename)
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return send_from_directory('../images', filename)
    except Exception as e:
        print(f"Error serving static file {filename}: {e}")
    
    return jsonify({'error': 'File not found'}), 404


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    try:
        parts = filename.split('/')
        if len(parts) == 2:
            fold_dir = parts[0]
            audio_file = parts[1]
            return send_from_directory(os.path.join(AUDIO_BASE_PATH, fold_dir), audio_file)
    except Exception as e:
        print(f"Error serving audio: {e}")
    
    return jsonify({'error': 'Audio file not found'}), 404


@app.route('/classify', methods=['POST'])
def classify():
    """Classify audio using improved model"""
    try:
        data = request.get_json()
        class_filter = data.get('class_filter', 'random')
        fold_filter = data.get('fold_filter', 'random')
        
        print(f"\n📊 Classification Request:")
        print(f"   Class Filter: {class_filter}")
        print(f"   Fold Filter: {fold_filter}")
        
        metadata = load_audio_metadata()
        clip_name, info, audio_path = get_random_audio_clip(class_filter, fold_filter, metadata)
        
        if clip_name is None:
            return jsonify({'error': 'No audio clips found matching the filters'}), 404
        
        print(f"   Selected: {clip_name}")
        print(f"   True Class: {info['class_name']}")
        
        # Classify using improved model
        result = MODEL_DATA.classify(audio_path)
        
        # Build response
        response = {
            'clip_id': clip_name,
            'true_class': info['class_name'],
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'is_correct': result['predicted_class'] == info['class_name'],
            'processing_time': result['processing_time'],
            'audio_path': f"audio/fold{info['fold']}/{clip_name}",
            'fold': info['fold'],
            'all_probabilities': result.get('all_probabilities', {})
        }
        
        print(f"   Predicted: {result['predicted_class']} ({result['confidence']:.1f}% confidence)")
        print(f"   Result: {'✓ CORRECT' if response['is_correct'] else '✗ INCORRECT'}")
        print(f"   Processing Time: {result['processing_time']:.3f}s\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"✗ Classification error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get model statistics"""
    try:
        # Load metadata from file
        with open('../models/improved_model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return jsonify(metadata)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_DATA is not None,
        'model_type': 'Improved Random Forest (72.52%)'
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting IMPROVED Sound Detector Demo Server")
    print("="*60)
    
    if not load_model():
        print("\n✗ Failed to load improved model. Exiting...")
        exit(1)
    
    print(f"\n📁 Audio Base Path: {AUDIO_BASE_PATH}")
    print(f"📄 CSV Path: {CSV_PATH}")
    
    print("\n" + "="*60)
    print("✓ Server ready with IMPROVED MODEL (72.52% accuracy)!")
    print("="*60)
    print("\n📡 Endpoints:")
    print("   GET  /              - Demo page")
    print("   GET  /demo_details.html - Details page")
    print("   POST /classify      - Classify audio (IMPROVED)")
    print("   GET  /stats         - Model statistics")
    print("   GET  /audio/<path>  - Serve audio files")
    print("   GET  /health        - Health check")
    print("\n🌐 Open browser to: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
