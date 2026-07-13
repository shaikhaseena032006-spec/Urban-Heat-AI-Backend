
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, HTTPException
import os
import requests
import pandas as pd
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
 
load_dotenv()
 
SENTINEL_CLIENT_ID = os.getenv("SENTINEL_CLIENT_ID")
SENTINEL_CLIENT_SECRET = os.getenv("SENTINEL_CLIENT_SECRET")
 
app = FastAPI(title="Delhi Urban Heat AI")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ======================
# Paths
# ======================
 
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "urban_heat_model.pkl"
CSV_PATH = BASE_DIR.parent / "outputs" / "dashboard_dataset.csv"
 
# ======================
# Load Model
# ======================
 
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Model loading failed: {e}")
 
 
# ======================
# City Bounding Boxes
# ======================
 
CITY_BBOX = {
    "Delhi": [76.84, 28.40, 77.35, 28.88],
    "Mumbai": [72.77, 18.89, 72.99, 19.28],
    "Hyderabad": [78.25, 17.20, 78.70, 17.60],
    "Bengaluru": [77.40, 12.80, 77.80, 13.20],
    "Chennai": [80.10, 12.85, 80.35, 13.20],
    "Kolkata": [88.20, 22.45, 88.55, 22.75]
}
 
 
# ======================
# Sentinel Hub Helpers
# ======================
 
def get_sentinel_token():
    url = "https://services.sentinel-hub.com/oauth/token"
 
    response = requests.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": SENTINEL_CLIENT_ID,
            "client_secret": SENTINEL_CLIENT_SECRET,
        },
    )
 
    response.raise_for_status()
 
    return response.json()["access_token"]
 
 
def get_satellite_index(city, index_name):
 
    bbox = CITY_BBOX.get(city)
 
    if not bbox:
        return {"error": "City not found"}
 
    if index_name == "ndvi":
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B04","B08"],
            output: { bands: 3 }
          };
        }
 
        function evaluatePixel(sample) {
          let ndvi =
            (sample.B08 - sample.B04) /
            (sample.B08 + sample.B04);
 
          if (ndvi < 0) return [120,80,40];
          if (ndvi < 0.2) return [255,255,0];
          if (ndvi < 0.5) return [0,255,0];
          return [0,120,0];
        }
        """
 
    elif index_name == "ndwi":
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B03","B08"],
            output: { bands: 3 }
          };
        }
 
        function evaluatePixel(sample) {
          let ndwi =
            (sample.B03 - sample.B08) /
            (sample.B03 + sample.B08);
 
          if (ndwi < 0)
            return [255,255,255];
 
          return [0,0,255];
        }
        """
 
    elif index_name == "ndbi":
        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B08","B11"],
            output: { bands: 3 }
          };
        }
 
        function evaluatePixel(sample) {
          let ndbi =
            (sample.B11 - sample.B08) /
            (sample.B11 + sample.B08);
 
          if (ndbi > 0)
            return [255,0,0];
 
          return [0,255,0];
        }
        """
 
    else:
        return {"error": "Unsupported index"}
 
    try:
        token = get_sentinel_token()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to get Sentinel Hub token: {e}"}
 
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/png",
        "Content-Type": "application/json"
    }
 
    body = {
        "input": {
            "bounds": {
                "bbox": bbox
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
        "evalscript": evalscript
    }
 
    try:
        response = requests.post(
            "https://services.sentinel-hub.com/api/v1/process",
            headers=headers,
            json=body
        )
    except requests.exceptions.RequestException as e:
        return {"error": f"Request to Sentinel Hub failed: {e}"}
 
    if response.status_code == 200:
        filename = f"{index_name}_{city}.png"
 
        with open(filename, "wb") as f:
            f.write(response.content)
 
        return {
            "status": 200,
            "file": filename
        }
 
    return {
        "status": response.status_code,
        "error": response.text
    }
 
 
# ======================
# Input Schemas
# ======================
 
class HeatInput(BaseModel):
    ndvi: float
    ndbi: float
    ndwi: float
    lat: float
    lon: float
 
 
class SimulationInput(BaseModel):
    ndvi: float
    ndbi: float
    ndwi: float
    lat: float
    lon: float
    intervention: str
 
 
class RecommendationInput(BaseModel):
    ndvi: float
    ndbi: float
 
 
# ======================
# Home API
# ======================
 
@app.get("/")
def home():
    return {
        "message": "Delhi Urban Heat AI Backend Running"
    }
 
 
# ======================
# Prediction API
# ======================
 
@app.post("/predict")
def predict(data: HeatInput):
 
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
 
    features = np.array([
        [
            data.ndvi,
            data.ndbi,
            data.ndwi,
            data.lat,
            data.lon
        ]
    ])
 
    prediction = float(model.predict(features)[0])
 
    if prediction < 32:
        risk = "Low"
    elif prediction < 38:
        risk = "Moderate"
    elif prediction < 44:
        risk = "High"
    else:
        risk = "Critical"
 
    return {
        "predicted_temperature": round(prediction, 2),
        "risk_level": risk
    }
 
 
# ======================
# Simulation API
# ======================
 
@app.post("/simulate")
def simulate(data: SimulationInput):
 
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
 
    current_features = np.array([
        [
            data.ndvi,
            data.ndbi,
            data.ndwi,
            data.lat,
            data.lon
        ]
    ])
 
    current_temp = model.predict(current_features)[0]
 
    ndvi = data.ndvi
    ndbi = data.ndbi
    ndwi = data.ndwi
 
    if data.intervention == "tree":
        ndvi += 0.20
 
    elif data.intervention == "cool_roof":
        ndbi -= 0.10
 
    elif data.intervention == "water":
        ndwi += 0.15
 
    elif data.intervention == "urban_greening":
        ndvi += 0.15
        ndbi -= 0.05
 
    else:
        raise HTTPException(status_code=400, detail="Unknown intervention type")
 
    future_features = np.array([
        [
            ndvi,
            ndbi,
            ndwi,
            data.lat,
            data.lon
        ]
    ])
 
    future_temp = model.predict(future_features)[0]
 
    return {
        "current_temp": round(float(current_temp), 2),
        "future_temp": round(float(future_temp), 2),
        "cooling_effect": round(float(current_temp - future_temp), 2),
        "intervention": data.intervention
    }
 
 
# ======================
# Recommendation API
# ======================
 
@app.post("/recommend")
def recommend(data: RecommendationInput):
 
    recommendations = []
 
    # Very low vegetation
    if data.ndvi < 0.10:
        recommendations.append("Mass Tree Plantation")
        recommendations.append("Urban Forest Development")
        recommendations.append("Green Corridors Along Roads")
 
    # Low vegetation
    elif data.ndvi < 0.25:
        recommendations.append("Urban Greening")
        recommendations.append("Rooftop Gardens")
        recommendations.append("Vertical Green Walls")
 
    # High built-up density
    if data.ndbi > 0.50:
        recommendations.append("Cool Roof Program")
        recommendations.append("Reflective Pavements")
        recommendations.append("Heat Resilient Urban Design")
 
    elif data.ndbi > 0.30:
        recommendations.append("Pocket Parks")
        recommendations.append("Shade Structures")
        recommendations.append("Solar Canopy Corridors")
 
    # Combined critical condition
    if data.ndvi < 0.20 and data.ndbi > 0.50:
        recommendations.append("Priority Heat Action Plan")
        recommendations.append("Emergency Cooling Centers")
        recommendations.append("Water Mist Systems")
 
    # Fallback
    if len(recommendations) == 0:
        recommendations.append("Monitor Environmental Conditions")
        recommendations.append("Maintain Existing Green Cover")
 
    return {
        "recommendation": " | ".join(recommendations)
    }
 
 
# ======================
# Dashboard / Analytics
# ======================
 
@app.get("/dashboard-data")
def dashboard_data():
 
    if not CSV_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="dashboard_dataset.csv not found"
        )
 
    df = pd.read_csv(CSV_PATH)
 
    return {
        "total_records": len(df),
        "avg_temperature": round(df["Predicted_LST"].mean(), 2),
        "max_temperature": round(df["Predicted_LST"].max(), 2),
        "min_temperature": round(df["Predicted_LST"].min(), 2)
    }
 
 
@app.get("/analytics")
def analytics():
 
    if not CSV_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="dashboard_dataset.csv not found"
        )
 
    df = pd.read_csv(CSV_PATH)
 
    return {
        "avg_temperature": round(df["Predicted_LST"].mean(), 2),
        "max_temperature": round(df["Predicted_LST"].max(), 2),
        "hotspots": int((df["Predicted_LST"] > 42).sum()),
        "high_risk_zones": int((df["Predicted_LST"] > 40).sum()),
        "vegetation_score": round(df["NDVI"].mean(), 2),
        "builtup_score": round(df["NDBI"].mean(), 2)
    }
 
 
# ======================
# NDVI / NDWI / NDBI APIs
# ======================
 
@app.get("/ndvi/{city}")
def get_city_ndvi(city: str):
 
    bbox = CITY_BBOX.get(city)
 
    if not bbox:
        raise HTTPException(
            status_code=404,
            detail=f"City '{city}' not found"
        )
 
    try:
        token = get_sentinel_token()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Sentinel Hub auth failed: {e}")
 
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
 
    payload = {
        "input": {
            "bounds": {
                "bbox": bbox
            },
            "data": [{
                "type": "sentinel-2-l2a"
            }]
        },
        "output": {
            "width": 256,
            "height": 256
        },
        "evalscript": """
        //VERSION=3
 
        function setup() {
          return {
            input: ["B04","B08"],
            output: { bands: 1 }
          };
        }
 
        function evaluatePixel(sample) {
          let ndvi =
            (sample.B08 - sample.B04) /
            (sample.B08 + sample.B04);
 
          return [ndvi];
        }
        """
    }
 
    try:
        response = requests.post(
            "https://services.sentinel-hub.com/api/v1/process",
            headers=headers,
            json=payload
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Sentinel Hub request failed: {e}")
 
    return {
        "city": city,
        "status": response.status_code
    }
 
 
@app.post("/ndvi")
def get_location_ndvi(data: dict):
 
    if "lat" not in data or "lon" not in data:
        raise HTTPException(status_code=400, detail="lat and lon are required")
 
    lat = data["lat"]
    lon = data["lon"]
 
    try:
        token = get_sentinel_token()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Sentinel Hub auth failed: {e}")
 
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
 
    payload = {
        "input": {
            "bounds": {
                "bbox": [
                    lon - 0.01,
                    lat - 0.01,
                    lon + 0.01,
                    lat + 0.01
                ]
            },
            "data": [
                {
                    "type": "sentinel-2-l2a"
                }
            ]
        },
        "evalscript": """
        //VERSION=3
 
        function setup() {
            return {
                input: ["B04","B08"],
                output: { bands: 1 }
            };
        }
 
        function evaluatePixel(sample) {
 
            let ndvi =
            (sample.B08 - sample.B04) /
            (sample.B08 + sample.B04);
 
            return [ndvi];
        }
        """
    }
 
    try:
        response = requests.post(
            "https://services.sentinel-hub.com/api/v1/process",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Sentinel Hub request failed: {e}")
 
    return {
        "status": response.status_code,
        "message": "NDVI request sent"
    }
 
 
@app.get("/ndwi/{city}")
def get_ndwi(city: str):
 
    result = get_satellite_index(city, "ndwi")
 
    return {
        "city": city,
        "status": result
    }
 
 
@app.get("/ndbi/{city}")
def get_ndbi(city: str):
 
    result = get_satellite_index(city, "ndbi")
 
    return {
        "city": city,
        "status": result
    }
 
 
# ======================
# Static City Overview Data
# ======================
 
@app.get("/city-data/{city}")
def city_data(city: str):
 
    cities = {
        "Delhi": {
            "avg_temp": 43,
            "risk": 85,
            "hotspots": [
                {"name": "Rohini", "lat": 28.74, "lng": 77.10},
                {"name": "Dwarka", "lat": 28.59, "lng": 77.04}
            ]
        },
        "Hyderabad": {
            "avg_temp": 40,
            "risk": 78,
            "hotspots": [
                {"name": "Gachibowli", "lat": 17.44, "lng": 78.35},
                {"name": "Madhapur", "lat": 17.45, "lng": 78.39}
            ]
        }
    }
 
    if city not in cities:
        raise HTTPException(
            status_code=404,
            detail="City not supported"
        )
 
    return cities[city]
 