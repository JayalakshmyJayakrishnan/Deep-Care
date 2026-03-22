import requests
import sys

url = "http://localhost:5000/api/upload"
filepath = r"d:\MINOR\pillcam_platform\uploads\pillcam_orig.jpg"

try:
    with open(filepath, 'rb') as f:
        files = {'file': f}
        print(f"Sending POST to {url}...")
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
