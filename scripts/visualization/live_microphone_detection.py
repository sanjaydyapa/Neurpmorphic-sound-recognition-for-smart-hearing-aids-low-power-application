"""
LIVE MICROPHONE INPUT MODULE
Real-time sound detection using neuromorphic principles

This module demonstrates the optional extension mentioned in the abstract:
"Extend to live microphone input for real-time testing"

Requirements: pyaudio (install with: pip install pyaudio)
"""

import numpy as np
import librosa
import sounddevice as sd
from sklearn.metrics import mean_squared_error
from datetime import datetime
import json
import queue
import threading

# Load the trained fingerprint
# You would normally save/load this from the main training
ENERGY_THRESHOLD = 0.05
MATCH_THRESHOLD = 4500
SAMPLE_RATE = 22050
CHUNK_DURATION = 0.25  # 250ms chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


class LiveNeuromorphicDetector:
    """
    Real-time neuromorphic sound detector for live microphone input.
    Implements event-driven processing for low-power operation.
    """
    
    def __init__(self, target_fingerprint, target_class, energy_thresh, match_thresh):
        self.target_fingerprint = target_fingerprint
        self.target_class = target_class
        self.energy_threshold = energy_thresh
        self.match_threshold = match_thresh
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.events = []
        
        print("╔═══════════════════════════════════════════════════════╗")
        print("║     LIVE NEUROMORPHIC SOUND DETECTOR STARTED          ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print(f"  Target Sound: {target_class}")
        print(f"  Sample Rate: {SAMPLE_RATE} Hz")
        print(f"  Chunk Size: {CHUNK_DURATION}s ({CHUNK_SAMPLES} samples)")
        print(f"  Energy Threshold: {energy_thresh}")
        print(f"  Match Threshold: {match_thresh}")
        print("  Status: LISTENING... (Press Ctrl+C to stop)\n")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio Stream Status: {status}")
        # Add audio data to queue for processing
        self.audio_queue.put(indata.copy())
    
    def process_audio_chunk(self, audio_chunk):
        """
        Process one chunk of audio using neuromorphic spike detection.
        """
        # Flatten audio if stereo
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
        
        # --- 1. Energy Spike Check ---
        rms_energy = np.sqrt(np.mean(audio_chunk**2))
        
        if rms_energy < self.energy_threshold:
            return None  # Too quiet, save power
        
        # --- 2. MFCC Pattern Matching ---
        try:
            mfccs = librosa.feature.mfcc(y=audio_chunk, sr=SAMPLE_RATE, n_mfcc=13)
            current_fingerprint = np.mean(mfccs, axis=1)
            
            # Calculate similarity
            similarity_error = mean_squared_error(self.target_fingerprint, current_fingerprint)
            
            if similarity_error < self.match_threshold:
                # SPIKE DETECTED!
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                event = {
                    'timestamp': timestamp,
                    'sound': self.target_class,
                    'energy': float(rms_energy),
                    'error': float(similarity_error)
                }
                self.events.append(event)
                
                print(f"🔴 [{timestamp}] {self.target_class.upper()} DETECTED! "
                      f"(Energy: {rms_energy:.3f}, Error: {similarity_error:.1f})")
                
                return event
        except Exception as e:
            # Handle edge cases (very short audio, etc.)
            pass
        
        return None
    
    def start_listening(self, duration=None):
        """
        Start listening to microphone input.
        
        Args:
            duration: How long to listen (seconds). None = listen until interrupted.
        """
        self.is_running = True
        
        try:
            # Start audio stream
            with sd.InputStream(callback=self.audio_callback,
                              channels=1,
                              samplerate=SAMPLE_RATE,
                              blocksize=CHUNK_SAMPLES):
                
                start_time = datetime.now()
                
                while self.is_running:
                    if duration and (datetime.now() - start_time).seconds >= duration:
                        break
                    
                    # Get audio chunk from queue
                    if not self.audio_queue.empty():
                        audio_chunk = self.audio_queue.get()
                        self.process_audio_chunk(audio_chunk)
        
        except KeyboardInterrupt:
            print("\n\n⏹ Stopping detector...")
        finally:
            self.is_running = False
            self.save_events()
    
    def save_events(self):
        """Save detected events to file"""
        if self.events:
            filename = f"live_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.events, f, indent=2)
            print(f"\n✓ {len(self.events)} events saved to {filename}")
        else:
            print("\n⚠ No events detected during session.")


def demo_live_detection():
    """
    Demo function showing how to use live detection.
    
    NOTE: You need to load the trained fingerprint from your training phase.
    For this demo, we'll create a dummy fingerprint.
    """
    
    print("\n" + "="*60)
    print("  LIVE MICROPHONE DETECTION DEMO")
    print("  (This is the optional extension from the abstract)")
    print("="*60 + "\n")
    
    print("⚠ IMPORTANT: This requires a trained fingerprint.")
    print("   Run the main training script first to generate fingerprints.")
    print("   Then modify this script to load your trained model.\n")
    
    # For demo purposes, create a dummy fingerprint
    # In practice, you would load this from your training phase
    dummy_fingerprint = np.random.randn(13) * 100
    
    # Create detector
    detector = LiveNeuromorphicDetector(
        target_fingerprint=dummy_fingerprint,
        target_class='siren',
        energy_thresh=ENERGY_THRESHOLD,
        match_thresh=MATCH_THRESHOLD
    )
    
    # Start listening for 30 seconds (or until Ctrl+C)
    print("🎤 Listening for 30 seconds... Make some noise!\n")
    detector.start_listening(duration=30)
    
    print("\n" + "="*60)
    print("  DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    print("\n⚠ NOTE: Live microphone detection requires:")
    print("   1. A trained fingerprint from the main detector")
    print("   2. sounddevice library: pip install sounddevice")
    print("   3. A working microphone\n")
    
    response = input("Do you want to run the live detection demo? (y/n): ")
    if response.lower() == 'y':
        try:
            demo_live_detection()
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("   Make sure sounddevice is installed: pip install sounddevice")
    else:
        print("\nDemo skipped. Use this module as a reference for implementing")
        print("live detection in your project.")
