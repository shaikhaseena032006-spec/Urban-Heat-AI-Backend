import requests
import os
from dotenv import load_dotenv
from pathlib import Path
 
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
 
print("ENV PATH =", env_path)
print("ENV EXISTS =", env_path.exists())
 
CLIENT_ID = os.getenv("SENTINEL_CLIENT_ID")
CLIENT_SECRET = os.getenv("SENTINEL_CLIENT_SECRET")
 
print("CLIENT_ID =", CLIENT_ID)
print("SECRET EXISTS =", CLIENT_SECRET is not None)
 
 
def get_token():
    response = requests.post(
        "https://services.sentinel-hub.com/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
    )
 
    print(response.status_code)
 
    if response.status_code != 200:
        print(response.text)
        return None
 
    return response.json()["access_token"]
 
 
def calculate_ndvi():
    city = "Delhi"
 
    token = get_token()
 
    if not token:
        print("Failed to get token")
        return
 
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/png",
        "Content-Type": "application/json",
    }
 
    body = {
        "input": {
            "bounds": {
                "bbox": [
                    77.1025,
                    28.7041,
                    77.3025,
                    28.9041
                ]
            },
            "data": [
                {
                    "type": "sentinel-2-l2a"
                }
            ]
        },
        "output": {
            "width": 512,
            "height": 512
        },
        "evalscript": """
        //VERSION=3
 
        function setup() {
          return {
            input: ["B04", "B08"],
            output: {
              bands: 3
            }
          };
        }
 
        function evaluatePixel(sample) {
 
          let ndvi =
            (sample.B08 - sample.B04) /
            (sample.B08 + sample.B04);
 
          if (ndvi < 0) {
            return [120, 80, 40];
          }
 
          if (ndvi < 0.2) {
            return [255, 255, 0];
          }
 
          if (ndvi < 0.5) {
            return [0, 255, 0];
          }
 
          return [0, 120, 0];
        }
        """
    }
 
    response = requests.post(
        "https://services.sentinel-hub.com/api/v1/process",
        headers=headers,
        json=body
    )
 
    print("STATUS =", response.status_code)
 
    if response.status_code == 200:
        with open(f"ndvi_{city}.png", "wb") as f:
            f.write(response.content)
 
        print("SUCCESS: ndvi_output.png created")
    else:
        print(response.text)
 
 
if __name__ == "__main__":
    calculate_ndvi()