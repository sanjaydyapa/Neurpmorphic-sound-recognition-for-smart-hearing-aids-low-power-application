"""
AUDIO VERIFICATION TOOL
Play audio files and show detection results to verify correctness
"""

import soundata
import librosa
import numpy as np
from neuromorphic_sound_detector_final import NeuromorphicDetector, create_fingerprint
import sounddevice as sd
import time

def play_and_verify_detection():
    """
    Play test audio files and show detection results
    """
    print("=" * 70)
    print("AUDIO VERIFICATION TOOL - Verify Detection Results")
    print("=" * 70)
    print()
    
    # Load dataset
    print("📥 Loading dataset...")
    dataset = soundata.initialize('urbansound8k', data_home='urbansound8k_data')
    
    # Create fingerprint
    print("🎯 Creating target fingerprint...")
    target_fingerprint = create_fingerprint(dataset, 'siren', 20)
    
    # Initialize detector
    ENERGY_THRESHOLD = 0.05
    MATCH_THRESHOLD = 4500
    detector = NeuromorphicDetector(target_fingerprint, ENERGY_THRESHOLD, MATCH_THRESHOLD, 'siren')
    
    # Get test clips
    print("🧪 Preparing test clips...")
    test_clips = [clip for clip in dataset.load_clips().values() 
                 if clip.class_label == 'siren'][20:23]
    non_target_clips = [clip for clip in dataset.load_clips().values() 
                       if clip.class_label in ['street_music', 'dog_bark']][:4]
    
    all_test_clips = []
    for clip in test_clips:
        all_test_clips.append({'clip': clip, 'expected': 'SIREN', 'is_target': True})
    for clip in non_target_clips:
        all_test_clips.append({'clip': clip, 'expected': clip.class_label.upper(), 'is_target': False})
    
    print()
    print("=" * 70)
    print("TESTING CLIPS - Listen and verify detections")
    print("=" * 70)
    print()
    
    results = []
    
    for i, test_info in enumerate(all_test_clips, 1):
        clip = test_info['clip']
        expected = test_info['expected']
        is_target = test_info['is_target']
        
        print(f"\n{'='*70}")
        print(f"TEST CLIP #{i} of {len(all_test_clips)}")
        print(f"{'='*70}")
        print(f"📄 File: {clip.clip_id}")
        print(f"🎯 Expected: {expected}")
        print(f"⏱️  Duration: {clip.duration:.2f} seconds")
        
        # Load and process audio
        audio, sr = librosa.load(clip.audio_path, sr=22050)
        chunk_size = int(0.25 * sr)
        
        # Clear previous events
        detector.events = []
        
        # Process audio
        detected = False
        detection_count = 0
        
        for j in range(0, len(audio) - chunk_size, chunk_size):
            chunk = audio[j:j + chunk_size]
            if detector.process_chunk(chunk, sr, j // chunk_size):
                detected = True
                detection_count += 1
        
        # Show results
        prediction = "SIREN DETECTED" if detected else "NO SIREN"
        correct = (detected and is_target) or (not detected and not is_target)
        
        print(f"\n🔍 DETECTION RESULT: {prediction}")
        print(f"   Detection events: {detection_count}")
        
        if correct:
            print(f"✅ CORRECT PREDICTION!")
        else:
            print(f"❌ INCORRECT PREDICTION!")
        
        results.append({
            'clip_num': i,
            'expected': expected,
            'predicted': 'SIREN' if detected else 'NOT SIREN',
            'correct': correct
        })
        
        # Play audio
        print(f"\n🔊 PLAYING AUDIO...")
        print("   (Close this or wait for audio to finish)")
        
        try:
            sd.play(audio, sr)
            sd.wait()  # Wait for playback to finish
            print("   ✅ Playback finished")
        except Exception as e:
            print(f"   ⚠️  Audio playback unavailable: {e}")
            print("   (This is normal if no audio device is available)")
        
        # Pause between clips
        if i < len(all_test_clips):
            print("\n⏸️  Press Ctrl+C to stop, or wait 2 seconds for next clip...")
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                print("\n\n🛑 Verification stopped by user.")
                break
    
    # Show summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print()
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    print(f"Total Clips Tested: {total_count}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Incorrect Predictions: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    print()
    
    print("Detailed Results:")
    print("-" * 70)
    for r in results:
        status = "✅ CORRECT" if r['correct'] else "❌ INCORRECT"
        print(f"  Clip #{r['clip_num']}: Expected={r['expected']:15s} Predicted={r['predicted']:15s} {status}")
    
    print()
    print("=" * 70)
    print("✅ VERIFICATION COMPLETE!")
    print("=" * 70)
    print()
    print("💡 Tips:")
    print("   • Listen carefully to each clip")
    print("   • Sirens have distinctive rising/falling tones")
    print("   • Non-siren sounds should be clearly different")
    print("   • Verify the detector makes sense for each clip")
    print()


if __name__ == "__main__":
    try:
        play_and_verify_detection()
    except KeyboardInterrupt:
        print("\n\n🛑 Verification stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: If you get audio playback errors, this is normal.")
        print("The tool still shows detection results for verification.")
