import requests
import json

print("Testing API endpoint...")
print("="*60)

try:
    # Exactly what demo.html sends
    url = "http://localhost:5000/api/detect"
    data = {"sound_class": "random"}
    headers = {"Content-Type": "application/json"}
    
    print(f"URL: {url}")
    print(f"Data: {data}")
    print(f"Headers: {headers}")
    print("\nSending request...")
    
    response = requests.post(url, json=data, headers=headers, timeout=30)
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"\nResponse Body:")
    print(json.dumps(response.json(), indent=2))
    
except requests.exceptions.Timeout:
    print("❌ Request timed out after 30 seconds")
except requests.exceptions.ConnectionError as e:
    print(f"❌ Connection error: {e}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
