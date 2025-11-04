"""
Flask Backend Server for Demo-Ready Ensemble Model
Uses the ensemble model with 77% accuracy and 917 stable features
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import random
import os
import sys
import time
import numpy as np
import librosa
import pickle
from scipy import stats

app = Flask(__name__)
CORS(app)

# Global variables
MODEL_DATA = None
AUDIO_BASE_PATH = "../urbansound8k_data/audio"
CSV_PATH = "../urbansound8k_data/metadata/UrbanSound8K.csv"


class StableFeatureExtractor:
    """Extract only stable, reliable features that work on ALL audio files"""
    
    def __init__(self, n_mfcc=40, n_mels=128):
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        
    def extract_features(self, audio_path):
        """Extract robust features with error handling"""
        try:
            # Load audio with fixed duration (3 seconds max)
            y, sr = librosa.load(audio_path, duration=3.0, sr=22050)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            features = []
            
            # 1. MFCC Features (most reliable) - 40 coefficients
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            features.extend([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),
                np.max(mfcc, axis=1),
                np.min(mfcc, axis=1)
            ])
            
            # 2. MFCC Deltas (velocity)
            mfcc_delta = librosa.feature.delta(mfcc)
            features.extend([
                np.mean(mfcc_delta, axis=1),
                np.std(mfcc_delta, axis=1)
            ])
            
            # 3. MFCC Delta-Deltas (acceleration)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features.extend([
                np.mean(mfcc_delta2, axis=1),
                np.std(mfcc_delta2, axis=1)
            ])
            
            # 4. Mel-Spectrogram (very stable) - 128 bands
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, 
                                                     n_fft=2048, hop_length=512)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features.extend([
                np.mean(mel_spec_db, axis=1),
                np.std(mel_spec_db, axis=1),
                np.max(mel_spec_db, axis=1),
                np.min(mel_spec_db, axis=1)
            ])
            
            # 5. Spectral Features (all stable)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.max(spectral_centroids),
                np.min(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.mean(spectral_contrast, axis=1),
                np.std(spectral_contrast, axis=1),
                np.mean(spectral_flatness),
                np.std(spectral_flatness)
            ])
            
            # 6. Chroma STFT (stable version, not CQT)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1),
                np.max(chroma, axis=1),
                np.min(chroma, axis=1)
            ])
            
            # 7. Zero Crossing Rate (very stable)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr),
                np.max(zcr),
                np.min(zcr)
            ])
            
            # 8. RMS Energy (very stable)
            rms = librosa.feature.rms(y=y)[0]
            features.extend([
                np.mean(rms),
                np.std(rms),
                np.max(rms),
                np.min(rms)
            ])
            
            # 9. Onset Strength (stable)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            features.extend([
                np.mean(onset_env),
                np.std(onset_env),
                np.max(onset_env)
            ])
            
            # 10. Statistical moments of waveform
            features.extend([
                np.mean(y),
                np.std(y),
                stats.skew(y),
                stats.kurtosis(y)
            ])
            
            # Flatten all features
            feature_vector = []
            for feat in features:
                if isinstance(feat, np.ndarray):
                    feature_vector.extend(feat)
                else:
                    feature_vector.append(feat)
            
            return np.array(feature_vector)
            
        except Exception as e:
            print(f"Warning: Error processing {audio_path}: {e}")
            # Return zero vector of EXACT dimension if extraction fails
            return np.zeros(917)  # Fixed dimension


class DemoReadyDetector:
    """Demo-ready sound detector using ensemble model"""
    
    def __init__(self, model_path):
        print(f"Loading demo-ready ensemble model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_extractor = model_data['feature_extractor']
        self.class_names = model_data['class_names']
        print(f"✓ Demo-ready ensemble model loaded successfully!")
        print(f"✓ Model type: Ensemble (XGBoost + RandomForest + ExtraTrees)")
        print(f"✓ Features: 917 stable features")
        print(f"✓ Classes: {len(self.class_names)}")
    
    def predict(self, audio_path):
        """
        Predict the class of an audio file
        Returns: (predicted_class, confidence, all_probabilities)
        """
        # Extract features using the saved feature extractor
        features = self.feature_extractor.extract_features(audio_path)
        
        # Handle extraction failures
        if features is None or len(features) != 917:
            print(f"Warning: Feature extraction issue for {audio_path}")
            if features is None:
                features = np.zeros(917)
            elif len(features) < 917:
                features = np.pad(features, (0, 917 - len(features)), 'constant')
            else:
                features = features[:917]
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        predicted_class = self.class_names[prediction]
        confidence = probabilities[prediction]
        
        # Create probability dict
        prob_dict = {self.class_names[i]: float(probabilities[i]) 
                    for i in range(len(self.class_names))}
        
        return predicted_class, confidence, prob_dict


def load_model():
    """Load the demo-ready ensemble model"""
    global MODEL_DATA
    try:
        model_path = '../trained_models/demo_ready_model.pkl'
        MODEL_DATA = DemoReadyDetector(model_path)
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
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
        
        print(f"✓ Loaded metadata for {len(metadata)} audio files")
        return metadata
    except Exception as e:
        print(f"✗ Error loading metadata: {e}")
        return {}


def get_random_audio_file(sound_class, audio_metadata):
    """Get a random audio file for the specified class"""
    if sound_class == 'random':
        candidates = list(audio_metadata.items())
    else:
        candidates = [(fname, meta) for fname, meta in audio_metadata.items() 
                     if meta['class_name'] == sound_class]
    
    if not candidates:
        return None
    
    filename, meta = random.choice(candidates)
    fold = meta['fold']
    audio_path = os.path.join(AUDIO_BASE_PATH, f"fold{fold}", filename)
    
    return audio_path, meta['class_name']


@app.route('/')
def index():
    """Serve the demo page"""
    return send_from_directory('.', 'demo.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('.', path)


@app.route('/audio/<path:audio_path>')
def serve_audio(audio_path):
    """Serve audio files from the dataset"""
    try:
        # audio_path format: "fold1/12345-1-0-0.wav"
        full_path = os.path.join('..', 'urbansound8k_data', audio_path)
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        return send_from_directory(directory, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/<path:filename>')
def serve_images(filename):
    """Serve image files (PNG) from images or visualizations folders"""
    try:
        # Check if it's a PNG file
        if filename.endswith('.png'):
            # Try images folder first
            if os.path.exists(os.path.join('..', 'images', filename)):
                return send_from_directory(os.path.join('..', 'images'), filename)
            # Try visualizations folder
            elif os.path.exists(os.path.join('..', 'visualizations', filename)):
                return send_from_directory(os.path.join('..', 'visualizations'), filename)
        # If not found or not a PNG, try static folder
        return send_from_directory('.', filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/visualizations/<path:filename>')
def serve_visualizations(filename):
    """Serve visualization files"""
    try:
        return send_from_directory(os.path.join('..', 'visualizations'), filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    API endpoint for sound detection
    Request: { "sound_class": "air_conditioner" | "random" | etc. }
    Response: { "success": true, "prediction": {...}, "ground_truth": "..." }
    """
    try:
        data = request.json
        sound_class = data.get('sound_class', 'random')
        
        # Get random audio file
        audio_metadata = load_audio_metadata()
        result = get_random_audio_file(sound_class, audio_metadata)
        
        if result is None:
            return jsonify({
                'success': False,
                'error': f'No audio files found for class: {sound_class}'
            })
        
        audio_path, ground_truth = result
        
        # Predict
        start_time = time.time()
        predicted_class, confidence, probabilities = MODEL_DATA.predict(audio_path)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Check if prediction is correct
        is_correct = (predicted_class == ground_truth)
        
        # Get relative path for audio serving (e.g., "fold1/12345-1-0-0.wav")
        relative_audio_path = os.path.relpath(audio_path, os.path.join('..', 'urbansound8k_data'))
        relative_audio_path = relative_audio_path.replace('\\', '/')  # Use forward slashes for URLs
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': probabilities
            },
            'ground_truth': ground_truth,
            'is_correct': is_correct,
            'inference_time_ms': round(inference_time, 2),
            'audio_file': os.path.basename(audio_path),
            'audio_path': relative_audio_path  # Add full path for audio player
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about the current model"""
    try:
        metadata_path = '../trained_models/demo_ready_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return jsonify({
            'success': True,
            'model_info': metadata
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/demo_ready_metadata.json', methods=['GET'])
def serve_metadata():
    """Serve the demo-ready model metadata JSON file"""
    try:
        metadata_path = '../trained_models/demo_ready_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 404


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DEMO-READY ENSEMBLE MODEL SERVER")
    print("="*60)
    
    # Load model
    if not load_model():
        print("Failed to load model. Exiting.")
        exit(1)
    
    print("\n" + "="*60)
    print("Server starting on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
