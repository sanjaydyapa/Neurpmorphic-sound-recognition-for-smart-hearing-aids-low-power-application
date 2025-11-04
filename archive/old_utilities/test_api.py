import requests
import json

# Test the fixed API
url = "http://localhost:5000/api/detect"

# Test 1: Random selection
print("🧪 Test 1: Random sound selection...")
response = requests.post(url, json={"sound_class": "random"})
print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"✅ SUCCESS!")
    print(f"   - Audio: {result['audio_file']}")
    print(f"   - True Class: {result['ground_truth']}")
    print(f"   - Predicted: {result['predicted_class']}")
    print(f"   - Confidence: {result['confidence']*100:.1f}%")
    print(f"   - Correct: {result['is_correct']}")
else:
    print(f"❌ FAILED: {response.text}")

print("\n" + "="*60 + "\n")

# Test 2: Specific class (siren - best performing)
print("🧪 Test 2: Siren detection (99.7% accuracy)...")
response = requests.post(url, json={"sound_class": "siren"})
print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"✅ SUCCESS!")
    print(f"   - Audio: {result['audio_file']}")
    print(f"   - True Class: {result['ground_truth']}")
    print(f"   - Predicted: {result['predicted_class']}")
    print(f"   - Confidence: {result['confidence']*100:.1f}%")
    print(f"   - Correct: {result['is_correct']}")
else:
    print(f"❌ FAILED: {response.text}")

print("\n" + "="*60 + "\n")
print("🎉 API tests complete! Try the demo page now:")
print("   👉 http://localhost:5000")
