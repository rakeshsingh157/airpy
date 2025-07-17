from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
import os
from utils import download_model

app = FastAPI()

# Check if model exists or download
if not os.path.exists("rural_aqi_model.pkl"):
    download_model()

try:
    model = joblib.load("rural_aqi_model.pkl")
except Exception as e:
    print("❌ Model load error:", e)
    model = None

@app.get("/")
def root():
    return {"message": "API is working ✅"}

@app.get("/predict/")
def predict(location: str = Query(...)):
    url = f"https://airbackend.vercel.app/api/air-quality/{location}"
    response = requests.get(url)
    json_data = response.json()

    if json_data['status'] != 'ok':
        return {"error": "API error"}

    data = json_data['data']

    try:
        temp = data['iaqi']['t']['v']
        humidity = data['iaqi']['h']['v']
        wind = data['iaqi']['w']['v']
        day = int(data['time']['s'][8:10])
        month = int(data['time']['s'][5:7])
    except KeyError as e:
        return {"error": f"Missing data: {e}"}

    input_df = pd.DataFrame([{
        "Temperature": temp,
        "Humidity": humidity,
        "WindSpeed": wind,
        "Day": day,
        "Month": month
    }])

    prediction = model.predict(input_df)[0]

    return {
        "location": data['city']['name'],
        "time": data['time']['s'],
        "temperature": temp,
        "humidity": humidity,
        "wind": wind,
        "predicted_rural_aqi": round(prediction, 2)
    }
