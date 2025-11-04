"""
SPIKING NEURON VISUALIZATION
Visualize the neuromorphic spike behavior during detection
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import librosa
import soundata
from neuromorphic_sound_detector_final import NeuromorphicDetector, create_fingerprint

def visualize_spike_train(audio_path, detector, save_path='spike_train.png'):
    """
    Create a raster plot showing spike times (like biological neurons)
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=22050)
    duration = len(audio) / sr
    
    # Process audio and collect spike events
    chunk_size = int(0.25 * sr)
    detector.events = []
    
    spike_times = []
    energy_values = []
    error_values = []
    chunk_times = []
    
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i + chunk_size]
        timestamp = i / sr
        
        is_spike = detector.process_chunk(chunk, sr, i // chunk_size)
        
        chunk_times.append(timestamp)
        
        if is_spike:
            spike_times.append(timestamp)
    
    # Get energy and error values from events
    for event in detector.events:
        energy_values.append(event.energy)
        if event.match_error is not None:
            error_values.append(event.match_error)
        else:
            error_values.append(np.nan)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # 1. Audio waveform
    times = np.arange(len(audio)) / sr
    axes[0].plot(times, audio, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title('Audio Waveform', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, duration)
    
    # Mark spike times
    for spike_time in spike_times:
        axes[0].axvline(spike_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # 2. Spike raster plot (like neuroscience papers)
    axes[1].eventplot(spike_times, colors='red', linewidths=3, linelengths=0.8)
    axes[1].set_ylabel('Neuron Firing', fontsize=12)
    axes[1].set_title('Spike Train (Neuromorphic Neuron Activity)', fontsize=14, fontweight='bold')
    axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_xlim(0, duration)
    
    # Add spike count
    axes[1].text(0.02, 0.5, f'{len(spike_times)} Spikes', 
                transform=axes[1].transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 3. Energy levels (Stage 1 gate)
    axes[2].plot(chunk_times, energy_values, 'go-', linewidth=2, markersize=5)
    axes[2].axhline(detector.energy_threshold, color='orange', linestyle='--', 
                   linewidth=2, label=f'Threshold = {detector.energy_threshold}')
    axes[2].fill_between(chunk_times, 0, energy_values, 
                        where=np.array(energy_values) > detector.energy_threshold,
                        color='green', alpha=0.3, label='Above Threshold')
    axes[2].set_ylabel('RMS Energy', fontsize=12)
    axes[2].set_title('Stage 1: Energy Gate', fontsize=14, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, duration)
    
    # 4. Pattern matching errors (Stage 2)
    valid_times = [t for t, e in zip(chunk_times, error_values) if not np.isnan(e)]
    valid_errors = [e for e in error_values if not np.isnan(e)]
    
    axes[3].plot(valid_times, valid_errors, 'bo-', linewidth=2, markersize=5)
    axes[3].axhline(detector.match_threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold = {detector.match_threshold}')
    axes[3].fill_between(valid_times, 0, valid_errors, 
                        where=np.array(valid_errors) < detector.match_threshold,
                        color='red', alpha=0.3, label='Match Detected')
    axes[3].set_ylabel('MFCC Error (MSE)', fontsize=12)
    axes[3].set_xlabel('Time (seconds)', fontsize=12)
    axes[3].set_title('Stage 2: Pattern Matching', fontsize=14, fontweight='bold')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(0, duration)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Spike train visualization saved to: {save_path}")
    plt.close()
    
    return spike_times, energy_values, error_values


def create_neuron_animation(audio_path, detector, save_path='neuron_animation.gif'):
    """
    Create animated visualization of neuron firing
    (Note: This creates a GIF which may take time to generate)
    """
    print("🎬 Creating neuron animation...")
    print("   This may take a minute...")
    
    # Load and process audio
    audio, sr = librosa.load(audio_path, sr=22050)
    chunk_size = int(0.25 * sr)
    detector.events = []
    
    # Process all chunks first
    events_data = []
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i + chunk_size]
        timestamp = i / sr
        is_spike = detector.process_chunk(chunk, sr, i // chunk_size)
        
        event = detector.events[-1]
        events_data.append({
            'time': timestamp,
            'energy': event.energy,
            'error': event.match_error if event.match_error else 0,
            'is_spike': is_spike
        })
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    def init():
        ax1.clear()
        ax2.clear()
        
        # Left: Neuron diagram
        ax1.set_xlim(-1, 11)
        ax1.set_ylim(-1, 11)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('Neuromorphic Neuron', fontsize=14, fontweight='bold')
        
        # Right: Spike train
        ax2.set_xlim(0, events_data[-1]['time'] if events_data else 1)
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Spikes', fontsize=12)
        ax2.set_title('Spike Output', fontsize=14, fontweight='bold')
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        
        return []
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        if frame >= len(events_data):
            return []
        
        event = events_data[frame]
        
        # Left panel: Neuron diagram
        ax1.set_xlim(-1, 11)
        ax1.set_ylim(-1, 11)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Neuromorphic Neuron (t={event["time"]:.2f}s)', 
                     fontsize=14, fontweight='bold')
        
        # Draw neuron body
        neuron_color = 'red' if event['is_spike'] else 'lightblue'
        neuron = Circle((5, 5), 2, color=neuron_color, ec='black', linewidth=2)
        ax1.add_patch(neuron)
        
        # Draw dendrites (inputs)
        for angle in [30, 90, 150, 210, 270, 330]:
            rad = np.radians(angle)
            x_start = 5 + 2 * np.cos(rad)
            y_start = 5 + 2 * np.sin(rad)
            x_end = 5 + 3.5 * np.cos(rad)
            y_end = 5 + 3.5 * np.sin(rad)
            ax1.plot([x_start, x_end], [y_start, y_end], 'k-', linewidth=2)
        
        # Draw axon (output)
        if event['is_spike']:
            ax1.arrow(7, 5, 2, 0, head_width=0.3, head_length=0.3, 
                     fc='red', ec='red', linewidth=3)
            ax1.text(9.5, 5.5, 'SPIKE!', fontsize=12, fontweight='bold', color='red')
        else:
            ax1.plot([7, 9], [5, 5], 'k-', linewidth=2)
        
        # Show energy and error values
        ax1.text(5, 1, f'Energy: {event["energy"]:.3f}', 
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax1.text(5, 0.2, f'Error: {event["error"]:.0f}', 
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
        
        # Right panel: Spike train
        ax2.set_xlim(0, events_data[-1]['time'])
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Spikes', fontsize=12)
        ax2.set_title('Spike Output', fontsize=14, fontweight='bold')
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        
        # Plot all spikes up to current time
        spike_times = [e['time'] for e in events_data[:frame+1] if e['is_spike']]
        if spike_times:
            ax2.eventplot(spike_times, colors='red', linewidths=3, linelengths=0.8)
        
        # Current time indicator
        ax2.axvline(event['time'], color='blue', linestyle='--', linewidth=2, alpha=0.5)
        
        return []
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=len(events_data), 
                                  interval=200, blit=True, repeat=True)
    
    # Save as GIF
    try:
        anim.save(save_path, writer='pillow', fps=5)
        print(f"✅ Animation saved to: {save_path}")
    except Exception as e:
        print(f"⚠️  Could not save animation: {e}")
        print("   Try installing: pip install pillow")
    
    plt.close()


def create_neuron_schematic():
    """
    Create a detailed schematic of the neuromorphic detection architecture
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'NEUROMORPHIC SPIKE-BASED DETECTION ARCHITECTURE',
           ha='center', fontsize=16, fontweight='bold')
    
    # Input
    ax.add_patch(Rectangle((0.5, 7), 1.5, 1, facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(1.25, 7.5, 'Audio\nInput', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.arrow(2.2, 7.5, 0.8, 0, head_width=0.2, head_length=0.15, fc='black', ec='black')
    
    # Stage 1: Energy Gate
    ax.add_patch(Rectangle((3.5, 6.5), 2.5, 2, facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(4.75, 8, 'STAGE 1', ha='center', fontsize=11, fontweight='bold')
    ax.text(4.75, 7.5, 'Energy Gate', ha='center', fontsize=10)
    ax.text(4.75, 7, 'RMS > 0.05?', ha='center', fontsize=9, style='italic')
    
    # Branch: Below threshold
    ax.arrow(4.75, 6.3, 0, -1, head_width=0.2, head_length=0.15, fc='red', ec='red', linewidth=2)
    ax.text(4.75, 5.8, 'NO', ha='center', fontsize=9, color='red', fontweight='bold')
    ax.add_patch(Rectangle((3.8, 4.8), 1.9, 0.8, facecolor='pink', edgecolor='red', linewidth=2))
    ax.text(4.75, 5.2, 'NO SPIKE\n(Power Save)', ha='center', va='center', 
           fontsize=9, fontweight='bold')
    
    # Arrow: Above threshold
    ax.arrow(6.2, 7.5, 0.8, 0, head_width=0.2, head_length=0.15, fc='black', ec='black')
    ax.text(6.6, 7.8, 'YES', ha='center', fontsize=9, fontweight='bold')
    
    # Stage 2: Pattern Match
    ax.add_patch(Rectangle((7.5, 6.5), 2.5, 2, facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.text(8.75, 8, 'STAGE 2', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.75, 7.5, 'Pattern Match', ha='center', fontsize=10)
    ax.text(8.75, 7, 'MSE < 4500?', ha='center', fontsize=9, style='italic')
    
    # Branch: No match
    ax.arrow(8.75, 6.3, 0, -1, head_width=0.2, head_length=0.15, fc='orange', ec='orange', linewidth=2)
    ax.text(8.75, 5.8, 'NO', ha='center', fontsize=9, color='orange', fontweight='bold')
    ax.add_patch(Rectangle((7.8, 4.8), 1.9, 0.8, facecolor='wheat', edgecolor='orange', linewidth=2))
    ax.text(8.75, 5.2, 'NO SPIKE\n(Not Target)', ha='center', va='center', 
           fontsize=9, fontweight='bold')
    
    # Arrow: Match found
    ax.arrow(10.2, 7.5, 0.8, 0, head_width=0.2, head_length=0.15, fc='green', ec='green', linewidth=2)
    ax.text(10.6, 7.8, 'YES', ha='center', fontsize=9, color='green', fontweight='bold')
    
    # Output: SPIKE!
    ax.add_patch(Circle((12, 7.5), 0.7, facecolor='red', edgecolor='black', linewidth=3))
    ax.text(12, 7.5, 'SPIKE!', ha='center', va='center', fontsize=12, 
           fontweight='bold', color='white')
    
    # Output arrow
    ax.arrow(12.8, 7.5, 0.7, 0, head_width=0.2, head_length=0.15, 
            fc='red', ec='red', linewidth=3)
    
    # Legend/Info box
    info_text = """
    NEUROMORPHIC PRINCIPLES:
    • Event-driven: Only process loud sounds
    • Spike-based: Binary firing (0 or 1)
    • Low power: ~80% chunks filtered
    • Fast: 5.47ms per second of audio
    """
    ax.text(7, 3, info_text, ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Stats box
    stats_text = """
    PERFORMANCE:
    Accuracy: 71.4%
    Real-time: 182x
    CPU: ~0.5%
    """
    ax.text(7, 1, stats_text, ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('neuron_architecture_schematic.png', dpi=300, bbox_inches='tight')
    print("✅ Architecture schematic saved to: neuron_architecture_schematic.png")
    plt.close()


def main():
    """
    Main function to demonstrate spike visualization
    """
    print("=" * 70)
    print("NEUROMORPHIC SPIKE VISUALIZATION")
    print("=" * 70)
    print()
    
    # Load dataset
    print("📥 Loading dataset...")
    dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
    
    # Create fingerprint
    print("🎯 Creating fingerprint...")
    target_fingerprint = create_fingerprint(dataset, 'siren', 20)
    
    # Initialize detector
    detector = NeuromorphicDetector(target_fingerprint, 0.05, 4500, 'siren')
    
    # Get a test clip
    test_clips = [c for c in dataset.load_clips().values() 
                 if c.class_label == 'siren'][20:21]
    
    if test_clips:
        print(f"🔊 Analyzing: {test_clips[0].clip_id}")
        print()
        
        # Create visualizations
        print("1️⃣  Creating spike train visualization...")
        spike_times, energies, errors = visualize_spike_train(
            test_clips[0].audio_path, 
            detector,
            'spike_train_visualization.png'
        )
        
        print(f"   ✅ Found {len(spike_times)} spikes")
        print()
        
        print("2️⃣  Creating neuron schematic...")
        create_neuron_schematic()
        print()
        
        print("3️⃣  Creating neuron animation (optional, takes time)...")
        response = input("   Create animation GIF? (y/n, default: n): ").strip().lower()
        if response == 'y':
            create_neuron_animation(
                test_clips[0].audio_path,
                detector,
                'neuron_firing_animation.gif'
            )
        else:
            print("   Skipped animation")
        
        print()
        print("=" * 70)
        print("✅ SPIKE VISUALIZATIONS COMPLETE!")
        print("=" * 70)
        print()
        print("📁 Generated files:")
        print("   • spike_train_visualization.png - Raster plot of neuron firing")
        print("   • neuron_architecture_schematic.png - System architecture")
        if response == 'y':
            print("   • neuron_firing_animation.gif - Animated neuron behavior")
        print()
        print("💡 These visualizations show:")
        print("   - When the neuromorphic neuron 'fires' (spikes)")
        print("   - Energy levels triggering Stage 1")
        print("   - Pattern matching errors in Stage 2")
        print("   - Event-driven processing behavior")
        print()


if __name__ == "__main__":
    main()
