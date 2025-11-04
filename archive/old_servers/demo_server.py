"""
Flask Backend Server for Neuromorphic Sound Detector Demo
Handles audio classification requests from the web interface
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import random
import os
import time
import numpy as np
import librosa

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Global variables
MODEL_DATA = None
AUDIO_BASE_PATH = "../urbansound8k_data/audio"
CSV_PATH = "../urbansound8k_data/metadata/UrbanSound8K.csv"

class MultiClassDetector:
    """Multi-class neuromorphic sound detector"""
    
    def __init__(self, model_data):
        self.sound_classes = model_data['sound_classes']
        self.fingerprints = {cls: np.array(model_data['fingerprints'][cls]) 
                           for cls in self.sound_classes}
        # Use default thresholds or load from model if available
        if 'optimal_thresholds' in model_data:
            self.energy_threshold = model_data['optimal_thresholds']['energy_threshold']
            self.match_threshold = model_data['optimal_thresholds']['match_threshold']
        else:
            self.energy_threshold = 0.01  # Default energy threshold
            self.match_threshold = 10000  # Default match threshold
    
    def extract_features(self, audio_path):
        """Extract MFCC features from audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=4.0)
            
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Average across time to get fingerprint
            fingerprint = np.mean(mfccs, axis=1)
            
            # Calculate energy
            energy = np.sqrt(np.mean(y**2))
            
            return fingerprint, energy
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None, 0
    
    def classify(self, audio_path):
        """Classify audio file"""
        start_time = time.time()
        
        # Extract features
        fingerprint, energy = self.extract_features(audio_path)
        
        if fingerprint is None or energy < self.energy_threshold:
            return {
                'predicted_class': 'silence',
                'confidence': 0,
                'processing_time': time.time() - start_time,
                'energy_gate_passed': False
            }
        
        # Find best match
        best_class = None
        best_score = float('inf')
        
        for cls in self.sound_classes:
            # Calculate MSE
            mse = np.mean((fingerprint - self.fingerprints[cls])**2)
            
            if mse < best_score:
                best_score = mse
                best_class = cls
        
        # Calculate confidence (inverse of MSE, normalized)
        confidence = max(0, min(100, (1 - best_score / self.match_threshold) * 100))
        
        processing_time = time.time() - start_time
        
        return {
            'predicted_class': best_class,
            'confidence': round(confidence, 2),
            'processing_time': round(processing_time, 3),
            'energy_gate_passed': True,
            'mse_score': round(best_score, 2)
        }


def load_model():
    """Load the multi-class model"""
    global MODEL_DATA
    try:
        with open('../models/multi_class_model_all_sounds.json', 'r') as f:
            MODEL_DATA = json.load(f)
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def load_audio_metadata():
    """Load audio metadata from CSV"""
    metadata = {}
    try:
        with open(CSV_PATH, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            
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
    
    # Filter clips
    valid_clips = []
    for clip_name, info in metadata.items():
        # Apply class filter
        if class_filter and class_filter != 'random':
            if info['class_name'] != class_filter:
                continue
        
        # Apply fold filter
        if fold_filter and fold_filter != 'random':
            if info['fold'] != int(fold_filter):
                continue
        
        # Check if file exists
        audio_path = os.path.join(AUDIO_BASE_PATH, f"fold{info['fold']}", clip_name)
        if os.path.exists(audio_path):
            valid_clips.append((clip_name, info, audio_path))
    
    if not valid_clips:
        return None, None, None
    
    # Select random clip
    clip_name, info, audio_path = random.choice(valid_clips)
    
    return clip_name, info, audio_path


@app.route('/')
def index():
    """Serve the demo page"""
    return send_from_directory('.', 'demo.html')


@app.route('/demo_details.html')
def details():
    """Serve the details page"""
    return send_from_directory('.', 'demo_details.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (JSON, images, etc.)"""
    try:
        # Check if it's a JSON file
        if filename.endswith('.json'):
            return send_from_directory('../models', filename)
        
        # Check if it's an image in visualizations folder
        if filename.startswith('visualizations/'):
            return send_from_directory('..', filename)
        
        # Check if it's an image in images folder
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            return send_from_directory('../images', filename)
            
    except Exception as e:
        print(f"Error serving static file {filename}: {e}")
    
    return jsonify({'error': 'File not found'}), 404


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files"""
    try:
        # Parse fold and filename
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
    """Classify a random audio clip based on filters"""
    try:
        data = request.get_json()
        class_filter = data.get('class_filter', 'random')
        fold_filter = data.get('fold_filter', 'random')
        
        print(f"\n📊 Classification Request:")
        print(f"   Class Filter: {class_filter}")
        print(f"   Fold Filter: {fold_filter}")
        
        # Load metadata
        metadata = load_audio_metadata()
        
        # Get random clip
        clip_name, info, audio_path = get_random_audio_clip(class_filter, fold_filter, metadata)
        
        if clip_name is None:
            return jsonify({
                'error': 'No audio clips found matching the filters'
            }), 404
        
        print(f"   Selected: {clip_name}")
        print(f"   True Class: {info['class_name']}")
        
        # Initialize detector
        detector = MultiClassDetector(MODEL_DATA)
        
        # Classify
        result = detector.classify(audio_path)
        
        # Build response
        response = {
            'clip_id': clip_name,
            'true_class': info['class_name'],
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'is_correct': result['predicted_class'] == info['class_name'],
            'processing_time': result['processing_time'],
            'audio_path': f"audio/fold{info['fold']}/{clip_name}",
            'fold': info['fold']
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
    """Get overall model statistics"""
    try:
        if MODEL_DATA is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        stats = {
            'overall_accuracy': MODEL_DATA['overall_accuracy'],
            'sound_classes': MODEL_DATA['sound_classes'],
            'class_accuracies': MODEL_DATA['class_accuracies'],
            'training_clips': MODEL_DATA['training_clips'],
            'test_clips': MODEL_DATA['test_clips']
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_DATA is not None
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Neuromorphic Sound Detector Demo Server")
    print("="*60)
    
    # Load model
    if not load_model():
        print("\n✗ Failed to load model. Exiting...")
        exit(1)
    
    print(f"\n📁 Audio Base Path: {AUDIO_BASE_PATH}")
    print(f"📄 CSV Path: {CSV_PATH}")
    
    # Check if paths exist
    if not os.path.exists(AUDIO_BASE_PATH):
        print(f"\n⚠️  Warning: Audio directory not found: {AUDIO_BASE_PATH}")
    
    if not os.path.exists(CSV_PATH):
        print(f"\n⚠️  Warning: CSV file not found: {CSV_PATH}")
    
    print("\n" + "="*60)
    print("✓ Server ready!")
    print("="*60)
    print("\n📡 Endpoints:")
    print("   GET  /              - Demo page")
    print("   GET  /demo_details.html - Details page")
    print("   POST /classify      - Classify audio")
    print("   GET  /stats         - Model statistics")
    print("   GET  /audio/<path>  - Serve audio files")
    print("   GET  /health        - Health check")
    print("\n🌐 Open browser to: http://localhost:5000")
    print("="*60 + "\n")
    
    # Start server
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
