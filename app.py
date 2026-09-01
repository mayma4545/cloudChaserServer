import sys
import numpy as np
import pandas as pd
import requests
import joblib
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# OpenWeather API Key from predict_live_openweather.py
API_KEY = "b464192b3768fca390669e440fb0a9bd"

# Load AI models on startup
try:
    rf_model = joblib.load("rain_prediction_model_15m.pkl")
    xgb_model = joblib.load("rain_prediction_xgboost_15m.pkl")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    rf_model = None
    xgb_model = None

# Dictionary to keep track of the last weather data for 15-minute differences
# Key: (lat, lon), Value: {'temp_c': ..., 'humidity_pct': ..., 'pressure_hpa': ...}
last_weather_data = {}

@app.route('/api/weather', methods=['GET'])
def get_weather():
    # Default to Masbate City coordinates if not provided
    lat = request.args.get('lat', default=12.3715, type=float)
    lon = request.args.get('lon', default=123.6247, type=float)

    # 1. Fetch current weather from OpenWeatherMap
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return jsonify({'error': f"Failed to fetch data from OpenWeather: {str(e)}"}), 500

    # 2. Parse OpenWeatherMap data
    temp_c = data['main']['temp']
    humidity_pct = data['main']['humidity']
    pressure_hpa = data['main']['pressure']
    
    # Wind speed is in m/s, convert to km/h for the model
    wind_speed_ms = data['wind'].get('speed', 0)
    wind_speed_kmh = wind_speed_ms * 3.6
    wind_direction_deg = data['wind'].get('deg', 0)
    
    cloud_cover_pct = data['clouds'].get('all', 0)
    
    # 3. Calculate Dew Point & Cloud Base Height
    # Approximation for Dew Point: Td = T - ((100 - RH)/5)
    dew_point_c = temp_c - ((100 - humidity_pct) / 5)
    
    # Cloud base height (m) = (temp_c - dew_point_c) * 125
    cloud_base_height_m = (temp_c - dew_point_c) * 125
    
    # Cloud thickness - we can map cloud cover to thickness as a proxy based on user's request
    cloud_thickness_pct = cloud_cover_pct

    # 4. Calculate 15m diffs (using state if available, else 0 for the first request)
    state_key = (round(lat, 4), round(lon, 4))
    if state_key in last_weather_data:
        prev_data = last_weather_data[state_key]
        temp_drop_15m = temp_c - prev_data['temp_c']
        humidity_spike_15m = humidity_pct - prev_data['humidity_pct']
        pressure_drop_15m = pressure_hpa - prev_data['pressure_hpa']
    else:
        temp_drop_15m = 0.0
        humidity_spike_15m = 0.0
        pressure_drop_15m = 0.0
        
    # Update state for the next request
    last_weather_data[state_key] = {
        'temp_c': temp_c,
        'humidity_pct': humidity_pct,
        'pressure_hpa': pressure_hpa
    }
    
    # 5. Calculate Wind U and V for the model
    wind_radians = np.radians(wind_direction_deg)
    wind_u = -wind_speed_kmh * np.sin(wind_radians)
    wind_v = -wind_speed_kmh * np.cos(wind_radians)
    
    # 6. Predict Rainfall Probability
    rainfall_percentage = 0.0
    if rf_model and xgb_model:
        # Create feature dataframe
        feature_cols = [
            "temp_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh",
            "wind_u", "wind_v", "temp_drop_15m", "humidity_spike_15m", 
            "pressure_drop_15m"
        ]
        
        X_live = pd.DataFrame([{
            "temp_c": temp_c,
            "humidity_pct": humidity_pct,
            "pressure_hpa": pressure_hpa,
            "wind_speed_kmh": wind_speed_kmh,
            "wind_u": wind_u,
            "wind_v": wind_v,
            "temp_drop_15m": temp_drop_15m,
            "humidity_spike_15m": humidity_spike_15m,
            "pressure_drop_15m": pressure_drop_15m
        }], columns=feature_cols)
        
        try:
            rf_probs = rf_model.predict_proba(X_live)[:, 1] * 100
            xgb_probs = xgb_model.predict_proba(X_live)[:, 1] * 100
            # Average the predictions from both models just like predict_live_openweather.py
            rainfall_percentage = (rf_probs[0] + xgb_probs[0]) / 2
        except Exception as e:
            print(f"Prediction error: {e}")
            
    # 7. Format Response
    response_data = {
        "temperature_c": round(temp_c, 2),
        "humidity_pct": round(humidity_pct, 2),
        "wind_speed_kmh": round(wind_speed_kmh, 2),
        "cloud_base_height_m": round(cloud_base_height_m, 2),
        "cloud_thickness_pct": round(cloud_thickness_pct, 2),
        "rainfall_probability_pct": round(rainfall_percentage, 2)
    }

    return jsonify(response_data)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running"})

if __name__ == '__main__':
    # Run the server on all available IP addresses (0.0.0.0) on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
