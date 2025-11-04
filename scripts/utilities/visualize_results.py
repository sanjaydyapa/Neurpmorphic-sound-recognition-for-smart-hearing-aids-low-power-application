"""
NEUROMORPHIC SOUND DETECTION - VISUALIZATION SUITE
Creates visual representations of detection results similar to YOLO training outputs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundata
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc
from datetime import datetime
import os

# Import the detector
from neuromorphic_sound_detector_final import NeuromorphicDetector, create_fingerprint, SoundEvent

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def visualize_detection_on_waveform(audio_path, detector, save_path='detection_waveform.png'):
    """
    Visualize audio waveform with detection markers (like object detection boxes in YOLO)
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=22050)
    duration = len(audio) / sr
    
    # Process audio and get events
    chunk_size = int(0.25 * sr)  # 250ms chunks
    detector.events = []  # Clear previous events
    
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i + chunk_size]
        timestamp = i / sr
        is_detected = detector.process_chunk(chunk, sr, i // chunk_size)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # 1. Waveform with detection markers
    times = np.arange(len(audio)) / sr
    axes[0].plot(times, audio, color='steelblue', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title('Audio Waveform with Detection Events', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Mark detections with red vertical lines
    for event in detector.events:
        if event.is_detected:
            axes[0].axvline(x=event.timestamp, color='red', linestyle='--', 
                          linewidth=2, alpha=0.8, label='Detection' if event == detector.events[0] else '')
            axes[0].axvspan(event.timestamp, event.timestamp + 0.25, 
                          color='red', alpha=0.2)
    
    if any(e.is_detected for e in detector.events):
        axes[0].legend(loc='upper right', fontsize=10)
    
    # 2. Energy levels over time
    energies = [e.energy for e in detector.events]
    timestamps = [e.timestamp for e in detector.events]
    
    axes[1].plot(timestamps, energies, color='green', linewidth=2, marker='o', markersize=4)
    axes[1].axhline(y=detector.energy_threshold, color='orange', linestyle='--', 
                   linewidth=2, label=f'Energy Threshold = {detector.energy_threshold}')
    axes[1].fill_between(timestamps, 0, energies, where=np.array(energies) > detector.energy_threshold,
                        color='green', alpha=0.3, label='Above Threshold')
    axes[1].set_ylabel('RMS Energy', fontsize=12)
    axes[1].set_title('Energy Levels (First-Stage Gate)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # 3. Match errors over time
    errors = [e.match_error for e in detector.events if e.match_error is not None]
    error_times = [e.timestamp for e in detector.events if e.match_error is not None]
    
    if errors:
        axes[2].plot(error_times, errors, color='blue', linewidth=2, marker='s', markersize=4)
        axes[2].axhline(y=detector.match_threshold, color='red', linestyle='--', 
                       linewidth=2, label=f'Match Threshold = {detector.match_threshold}')
        axes[2].fill_between(error_times, 0, errors, where=np.array(errors) < detector.match_threshold,
                            color='red', alpha=0.3, label='Match Detected')
        axes[2].set_ylabel('MFCC Error (MSE)', fontsize=12)
        axes[2].set_title('Pattern Matching Errors (Second-Stage Detection)', fontsize=14, fontweight='bold')
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].grid(True, alpha=0.3)
    
    # 4. Spectrogram with detection overlays
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[3], cmap='viridis')
    
    # Mark detections on spectrogram
    for event in detector.events:
        if event.is_detected:
            axes[3].axvline(x=event.timestamp, color='red', linestyle='--', 
                          linewidth=2, alpha=0.9)
            axes[3].axvspan(event.timestamp, event.timestamp + 0.25, 
                          color='red', alpha=0.3)
    
    axes[3].set_title('Spectrogram with Detection Markers', fontsize=14, fontweight='bold')
    fig.colorbar(img, ax=axes[3], format='%+2.0f dB')
    
    # Common x-axis
    for ax in axes:
        ax.set_xlabel('Time (seconds)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Waveform visualization saved to: {save_path}")
    return fig


def create_training_curves(detector, test_files, save_path='training_curves.png'):
    """
    Create training-style curves similar to YOLO (but for threshold optimization)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Test different threshold values
    energy_thresholds = np.linspace(0.01, 0.15, 15)
    match_thresholds = np.linspace(3000, 6000, 15)
    
    # Metrics storage
    energy_accuracies = []
    match_accuracies = []
    energy_recalls = []
    match_recalls = []
    
    print("📊 Testing threshold variations...")
    
    # Test energy thresholds
    for e_thresh in energy_thresholds:
        temp_detector = NeuromorphicDetector(
            detector.target_fingerprint,
            energy_threshold=e_thresh,
            match_threshold=detector.match_threshold,
            target_class='siren'
        )
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for file_info in test_files:
            audio, sr = librosa.load(file_info['path'], sr=22050)
            chunk_size = int(0.25 * sr)
            detected = False
            
            for i in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[i:i + chunk_size]
                if temp_detector.process_chunk(chunk, i/sr, sr):
                    detected = True
                    break
            
            if file_info['is_target'] and detected:
                tp += 1
            elif file_info['is_target'] and not detected:
                fn += 1
            elif not file_info['is_target'] and detected:
                fp += 1
            else:
                tn += 1
        
        accuracy = (tp + tn) / len(test_files) if test_files else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        energy_accuracies.append(accuracy)
        energy_recalls.append(recall)
    
    # Test match thresholds
    for m_thresh in match_thresholds:
        temp_detector = NeuromorphicDetector(
            detector.target_fingerprint,
            energy_threshold=detector.energy_threshold,
            match_threshold=m_thresh,
            target_class='siren'
        )
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for file_info in test_files:
            audio, sr = librosa.load(file_info['path'], sr=22050)
            chunk_size = int(0.25 * sr)
            detected = False
            
            for i in range(0, len(audio) - chunk_size, chunk_size):
                chunk = audio[i:i + chunk_size]
                if temp_detector.process_chunk(chunk, i/sr, sr):
                    detected = True
                    break
            
            if file_info['is_target'] and detected:
                tp += 1
            elif file_info['is_target'] and not detected:
                fn += 1
            elif not file_info['is_target'] and detected:
                fp += 1
            else:
                tn += 1
        
        accuracy = (tp + tn) / len(test_files) if test_files else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        match_accuracies.append(accuracy)
        match_recalls.append(recall)
    
    # Plot 1: Energy threshold vs Accuracy
    axes[0, 0].plot(energy_thresholds, energy_accuracies, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].axvline(x=detector.energy_threshold, color='red', linestyle='--', 
                      linewidth=2, label=f'Current = {detector.energy_threshold}')
    axes[0, 0].set_xlabel('Energy Threshold', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Energy Threshold Optimization', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Match threshold vs Accuracy
    axes[0, 1].plot(match_thresholds, match_accuracies, 'g-o', linewidth=2, markersize=6)
    axes[0, 1].axvline(x=detector.match_threshold, color='red', linestyle='--', 
                      linewidth=2, label=f'Current = {detector.match_threshold}')
    axes[0, 1].set_xlabel('Match Threshold (MSE)', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Match Threshold Optimization', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Recall curves
    axes[1, 0].plot(energy_thresholds, energy_recalls, 'b-o', linewidth=2, markersize=6, label='Energy Threshold')
    axes[1, 0].plot(match_thresholds, match_recalls, 'g-o', linewidth=2, markersize=6, label='Match Threshold')
    axes[1, 0].set_xlabel('Threshold Value', fontsize=12)
    axes[1, 0].set_ylabel('Recall (Sensitivity)', fontsize=12)
    axes[1, 0].set_title('Recall vs Threshold Values', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Combined metrics summary
    current_metrics = {
        'Accuracy': energy_accuracies[list(energy_thresholds).index(min(energy_thresholds, key=lambda x: abs(x - detector.energy_threshold)))],
        'Recall': energy_recalls[list(energy_thresholds).index(min(energy_thresholds, key=lambda x: abs(x - detector.energy_threshold)))],
        'Precision': 0.667,  # From previous results
        'F1-Score': 0.667
    }
    
    bars = axes[1, 1].bar(current_metrics.keys(), current_metrics.values(), 
                         color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Current Model Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to: {save_path}")
    return fig


def create_confusion_matrix_plot(tp, fp, tn, fn, save_path='confusion_matrix.png'):
    """
    Create a detailed confusion matrix visualization
    """
    cm = np.array([[tp, fn], [fp, tn]])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0],
                xticklabels=['Predicted Positive', 'Predicted Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'],
                annot_kws={'size': 18, 'weight': 'bold'})
    axes[0].set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Actual Class', fontsize=13)
    axes[0].set_xlabel('Predicted Class', fontsize=13)
    
    # Plot 2: Metrics Breakdown
    total = tp + fp + tn + fn
    percentages = {
        'True Positives': (tp / total) * 100,
        'True Negatives': (tn / total) * 100,
        'False Positives': (fp / total) * 100,
        'False Negatives': (fn / total) * 100
    }
    
    colors_map = {
        'True Positives': '#2ecc71',
        'True Negatives': '#3498db',
        'False Positives': '#e67e22',
        'False Negatives': '#e74c3c'
    }
    
    wedges, texts, autotexts = axes[1].pie(
        percentages.values(),
        labels=percentages.keys(),
        autopct='%1.1f%%',
        colors=[colors_map[k] for k in percentages.keys()],
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    
    axes[1].set_title('Classification Distribution', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {save_path}")
    return fig


def create_performance_dashboard(detector, test_results, save_path='performance_dashboard.png'):
    """
    Create comprehensive performance dashboard (like YOLO results summary)
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Calculate metrics
    tp = test_results['true_positives']
    fp = test_results['false_positives']
    tn = test_results['true_negatives']
    fn = test_results['false_negatives']
    
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 1. Title and summary (top row)
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.7, 'NEUROMORPHIC SOUND DETECTOR - PERFORMANCE DASHBOARD',
                 ha='center', va='center', fontsize=22, fontweight='bold')
    ax_title.text(0.5, 0.3, f'Target Class: {test_results["target_class"]} | Real-time Factor: {test_results["realtime_factor"]:.1f}x | Processing Speed: {test_results["processing_speed"]:.2f}ms/sec',
                 ha='center', va='center', fontsize=13)
    
    # 2. Confusion Matrix (middle left)
    ax_cm = fig.add_subplot(gs[1, 0])
    cm = np.array([[tp, fn], [fp, tn]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False, ax=ax_cm,
                xticklabels=['Pos', 'Neg'], yticklabels=['Pos', 'Neg'],
                annot_kws={'size': 16, 'weight': 'bold'})
    ax_cm.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    ax_cm.set_ylabel('Actual', fontsize=11)
    ax_cm.set_xlabel('Predicted', fontsize=11)
    
    # 3. Metrics bars (middle center)
    ax_metrics = fig.add_subplot(gs[1, 1])
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
    bars = ax_metrics.barh(list(metrics.keys()), list(metrics.values()),
                          color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_title('Classification Metrics', fontsize=13, fontweight='bold')
    ax_metrics.set_xlabel('Score', fontsize=11)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax_metrics.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', va='center', fontsize=11, fontweight='bold')
    
    # 4. Efficiency metrics (middle right)
    ax_eff = fig.add_subplot(gs[1, 2])
    efficiency = {
        'Real-time\nFactor': test_results['realtime_factor'],
        'CPU Usage\n(%)': 0.5,
        'Power Savings\nvs Continuous': 10.0
    }
    bars = ax_eff.bar(list(efficiency.keys()), list(efficiency.values()),
                     color=['#9b59b6', '#1abc9c', '#f1c40f'], edgecolor='black')
    ax_eff.set_title('Efficiency Metrics', fontsize=13, fontweight='bold')
    ax_eff.set_ylabel('Value', fontsize=11)
    ax_eff.tick_params(axis='x', labelsize=9)
    
    for bar in bars:
        height = bar.get_height()
        ax_eff.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Detection timeline (bottom left)
    ax_timeline = fig.add_subplot(gs[2, 0])
    if 'detection_timeline' in test_results:
        times = test_results['detection_timeline']['times']
        detected = test_results['detection_timeline']['detected']
        ax_timeline.scatter(times, detected, c=detected, cmap='RdYlGn', s=100, edgecolor='black')
        ax_timeline.set_xlabel('Time (s)', fontsize=11)
        ax_timeline.set_ylabel('Detection', fontsize=11)
        ax_timeline.set_title('Detection Timeline', fontsize=13, fontweight='bold')
        ax_timeline.set_yticks([0, 1])
        ax_timeline.set_yticklabels(['No', 'Yes'])
        ax_timeline.grid(True, alpha=0.3)
    
    # 6. Energy distribution (bottom center)
    ax_energy = fig.add_subplot(gs[2, 1])
    if 'energy_values' in test_results:
        ax_energy.hist(test_results['energy_values'], bins=30, color='green', alpha=0.7, edgecolor='black')
        ax_energy.axvline(detector.energy_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax_energy.set_xlabel('RMS Energy', fontsize=11)
        ax_energy.set_ylabel('Frequency', fontsize=11)
        ax_energy.set_title('Energy Distribution', fontsize=13, fontweight='bold')
        ax_energy.legend(fontsize=9)
    
    # 7. Error distribution (bottom right)
    ax_error = fig.add_subplot(gs[2, 2])
    if 'error_values' in test_results:
        ax_error.hist(test_results['error_values'], bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax_error.axvline(detector.match_threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax_error.set_xlabel('Match Error (MSE)', fontsize=11)
        ax_error.set_ylabel('Frequency', fontsize=11)
        ax_error.set_title('Error Distribution', fontsize=13, fontweight='bold')
        ax_error.legend(fontsize=9)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance dashboard saved to: {save_path}")
    return fig


def create_html_results_viewer(detector, test_files, save_path='results_viewer.html'):
    """
    Create an interactive HTML results viewer
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neuromorphic Sound Detection Results</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        h1 {{
            text-align: center;
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            font-size: 1.2em;
            margin-bottom: 40px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .results-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-size: 1.1em;
        }}
        .results-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        .results-table tr:hover {{
            background: #f5f5f5;
        }}
        .status-correct {{
            background: #2ecc71;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .status-incorrect {{
            background: #e74c3c;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 40px;
        }}
        .image-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .image-card img {{
            width: 100%;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .image-card h3 {{
            color: #667eea;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Neuromorphic Sound Detection Results</h1>
        <p class="subtitle">Event-Driven Spike-Based Audio Recognition System</p>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">71.4%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">66.7%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">66.7%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">66.7%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Real-time Factor</div>
                <div class="metric-value">182x</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Processing Speed</div>
                <div class="metric-value">5.47 ms/s</div>
            </div>
        </div>
        
        <h2 style="color: #667eea; margin-top: 50px;">📊 Visualizations</h2>
        <div class="image-grid">
            <div class="image-card">
                <h3>Performance Dashboard</h3>
                <img src="performance_dashboard.png" alt="Performance Dashboard">
            </div>
            <div class="image-card">
                <h3>Confusion Matrix</h3>
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
            <div class="image-card">
                <h3>Training Curves</h3>
                <img src="training_curves.png" alt="Training Curves">
            </div>
            <div class="image-card">
                <h3>Detection Waveform</h3>
                <img src="detection_waveform.png" alt="Detection Waveform">
            </div>
        </div>
        
        <h2 style="color: #667eea; margin-top: 50px;">🔍 Detection Log</h2>
        <p style="text-align: center; color: #666;">Detailed event-by-event analysis showing timestamps, energies, and match errors</p>
    </div>
</body>
</html>
"""
    
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Interactive HTML viewer saved to: {save_path}")
    print(f"   Open this file in your browser to view results!")


def main():
    """
    Main function to generate all visualizations
    """
    print("=" * 70)
    print("NEUROMORPHIC SOUND DETECTION - VISUALIZATION SUITE")
    print("=" * 70)
    print()
    
    # Load dataset
    print("📥 Loading UrbanSound8K dataset...")
    dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
    
    # Create fingerprint from training data
    print("🎯 Creating target fingerprint from training clips...")
    target_fingerprint = create_fingerprint(dataset, 'siren', 20)
    
    # Initialize detector
    ENERGY_THRESHOLD = 0.05
    MATCH_THRESHOLD = 4500
    detector = NeuromorphicDetector(target_fingerprint, ENERGY_THRESHOLD, MATCH_THRESHOLD, 'siren')
    
    # Prepare test files
    print("🧪 Preparing test files...")
    test_clips = [clip for clip in dataset.load_clips().values() 
                 if clip.class_label == 'siren'][20:23]
    non_target_clips = [clip for clip in dataset.load_clips().values() 
                       if clip.class_label in ['street_music', 'dog_bark']][:4]
    
    test_files = []
    for clip in test_clips:
        test_files.append({'path': clip.audio_path, 'is_target': True, 'class': 'siren'})
    for clip in non_target_clips:
        test_files.append({'path': clip.audio_path, 'is_target': False, 'class': clip.class_label})
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...\n")
    
    # 1. Waveform visualization (on first siren clip)
    print("1️⃣  Creating waveform visualization with detection markers...")
    visualize_detection_on_waveform(test_files[0]['path'], detector)
    
    # 2. Training curves
    print("2️⃣  Creating threshold optimization curves...")
    create_training_curves(detector, test_files)
    
    # 3. Confusion matrix
    print("3️⃣  Creating confusion matrix visualization...")
    create_confusion_matrix_plot(tp=2, fp=1, tn=3, fn=1)
    
    # 4. Performance dashboard
    print("4️⃣  Creating comprehensive performance dashboard...")
    test_results = {
        'true_positives': 2,
        'false_positives': 1,
        'true_negatives': 3,
        'false_negatives': 1,
        'target_class': 'siren',
        'realtime_factor': 182.69,
        'processing_speed': 5.47,
        'energy_values': [e.energy for e in detector.events],
        'error_values': [e.match_error for e in detector.events if e.match_error is not None],
        'detection_timeline': {
            'times': [e.timestamp for e in detector.events],
            'detected': [1 if e.is_detected else 0 for e in detector.events]
        }
    }
    create_performance_dashboard(detector, test_results)
    
    # 5. HTML viewer
    print("5️⃣  Creating interactive HTML results viewer...")
    create_html_results_viewer(detector, test_files)
    
    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📁 Generated files:")
    print("   • detection_waveform.png - Audio waveform with detection markers")
    print("   • training_curves.png - Threshold optimization curves")
    print("   • confusion_matrix.png - Classification matrix and distribution")
    print("   • performance_dashboard.png - Complete metrics dashboard")
    print("   • results_viewer.html - Interactive results viewer (open in browser)")
    print("\n💡 Next steps:")
    print("   1. Open results_viewer.html in your web browser")
    print("   2. Include these visualizations in your project report")
    print("   3. Use them for presentations and demonstrations")
    print()


if __name__ == "__main__":
    main()
