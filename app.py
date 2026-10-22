"""
=============================================================================
Cloud Chaser - Headless Data Gathering Server (dataGatheringServer.py)
=============================================================================
Runs completely in the background without displaying any GUI/video window
to drastically reduce CPU & GPU memory consumption while continuously extracting:

  1. Sky Telemetry directly from live video / RTSP stream (Headless):
     - Cloud Thickness (%)
     - Cloud Coverage (%)
     - Cloud Density (% Cloud-only thickness)
     - Scene Light (% Mean Brightness)
     - YOLO cloud detection (clouds/best (5).pt)
     - ResNet CCSN_v2 cloud genus classifier (resnet_cloud_type_model.pth)
     - Dominant Cloud Genus (Area- & confidence-weighted voting)
     - Meteorological Outlook (Atmospheric Level & Rain Outlook)

  2. Ambient & Local Sensor Telemetry:
     - Local ESP32 SHT45 Temperature (°C) & Humidity (%RH)
     - Ambient OpenWeather parameters (Pressure, Wind vector U/V, 1h rain)
     - Cloud Base Height (m) calculated from temperature & dew point

  3. Intelligent Dual-Ensemble Rain Prediction Computation:
     - Sensor ML Ensemble: Random Forest (15m) + XGBoost (15m) average
     - Meteorological Visual Outlook Score: Cloud Genus Risk + Optical Thickness
     - Final Adjusted Rain Prediction: Blends ML thermodynamic sensors with
       visual cloud genus observations, applying physical guardrails.

  4. Continuous CSV Persistence with exact timestamps:
     - Appends all data to gathered_weather_data.csv
     - Web UI Dashboard and JSON REST API
=============================================================================
"""

import os
import sys
import time
import json
import csv
import threading
import argparse
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import requests
import joblib
from flask import Flask, request, jsonify, send_file, render_template_string, Response
from flask_cors import CORS

def load_local_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(name.strip(), value)

load_local_env()

# Add cloudThicknessTest to module path for core computer vision functions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check if running from cloudChaserServer/ or project root
if os.path.basename(BASE_DIR) == "cloudChaserServer":
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
else:
    PROJECT_ROOT = BASE_DIR

CLOUD_TEST_DIR = os.path.join(PROJECT_ROOT, "cloudThicknessTest")
if CLOUD_TEST_DIR not in sys.path:
    sys.path.insert(0, CLOUD_TEST_DIR)

from live_cloud_thickness import (
    get_roi_mask,
    calculate_thickness,
    load_yolo_model,
    load_cloud_type_classifier,
    classify_cloud_patch,
    get_cloud_details,
    DEFAULT_CLOUD_INFO
)

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "gathered_weather_data.csv")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Default RTSP Stream URL
DEFAULT_RTSP_URL = os.getenv("RTSP_URL", "")

# Masbate City coordinates (default)
DEFAULT_LAT = 12.3715
DEFAULT_LON = 123.6247

# Path to local sensor data written by ESP32 / cloudChaserServer
LOCAL_SENSOR_JSON = os.path.join(PROJECT_ROOT, "cloudChaserServer", "sensor_data.json")
if not os.path.exists(LOCAL_SENSOR_JSON):
    LOCAL_SENSOR_JSON = os.path.join(PROJECT_ROOT, "sensor_data.json")

# =============================================================================
# 2. METEOROLOGICAL GENUS RAIN PRIORS (WMO 11 Classes)
# =============================================================================
GENUS_RAIN_PRIOR = {
    "Cb": 92.0,  # Cumulonimbus: Thunderstorm, flash rain, downpour
    "Ns": 88.0,  # Nimbostratus: Continuous steady rain
    "As": 52.0,  # Altostratus: Thickening leads to rain/snow
    "St": 38.0,  # Stratus: Drizzle, mist
    "Sc": 32.0,  # Stratocumulus: Overcast, occasional light rain
    "Cu": 18.0,  # Cumulus: Fair weather / developing
    "Ac": 15.0,  # Altocumulus: Mostly fair, rare virga
    "Cs": 8.0,   # Cirrostratus: Halo, dry
    "Cc": 5.0,   # Cirrocumulus: Rare sprinkles
    "Ci": 3.0,   # Cirrus: Dry upper air, fair
    "Ct": 0.0,   # Contrail: Dry artificial trail
    "Clear": 0.0 # Clear Sky
}

# =============================================================================
# 3. FLASK APP & STATE STORAGE
# =============================================================================
app = Flask(__name__)
CORS(app)

# In-memory storage
latest_gathered_data = None
recent_history = []
latest_direct_sensor = None
prev_weather_state = None

# Thread-safe Sky Telemetry (updated by headless background video worker)
sky_telemetry_lock = threading.Lock()
latest_sky_telemetry = {
    "cloud_thickness_pct": 0.0,
    "cloud_coverage_pct": 0.0,
    "cloud_density_pct": 0.0,
    "scene_light_pct": 0.0,
    "num_clouds_detected": 0,
    "dominant_cloud_genus": "Standby",
    "dominant_cloud_code": "Clear",
    "atmospheric_level": "Clear Atmosphere",
    "rain_outlook": "Standby / Initializing",
    "visual_meteo_score_pct": 0.0,
    "stream_status": "Starting background stream...",
    "last_updated": "Never"
}

latest_frame = None
latest_frame_lock = threading.Lock()

# =============================================================================
# 4. LOAD MACHINE LEARNING & COMPUTER VISION MODELS
# =============================================================================
def find_file(filename):
    candidates = [
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(PROJECT_ROOT, "cloudChaserServer", filename),
        os.path.join(PROJECT_ROOT, "cloudThicknessTest", filename),
        os.path.join(PROJECT_ROOT, "clouds", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

print("[INFO] Loading ML Weather & Vision Models...", flush=True)

# 1. Random Forest (15m)
rf_path = find_file("rain_prediction_model_15m.pkl")
rf_model = None
if rf_path:
    try:
        rf_model = joblib.load(rf_path)
        print(f"[OK] Random Forest Model loaded from: {rf_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to load Random Forest model: {e}", flush=True)

# 2. XGBoost (15m)
xgb_path = find_file("rain_prediction_xgboost_15m.pkl")
xgb_model = None
if xgb_path:
    try:
        xgb_model = joblib.load(xgb_path)
        print(f"[OK] XGBoost Model loaded from: {xgb_path}", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to load XGBoost model: {e}", flush=True)

# 3. YOLO Cloud Model
yolo_path = find_file(os.path.join("clouds", "best (5).pt")) or find_file("best (5).pt")
yolo_model = None
if yolo_path:
    yolo_model, _ = load_yolo_model(yolo_path)

# 4. ResNet Cloud Genus Classifier (CCSN_v2 11 Classes)
resnet_path = find_file("resnet_cloud_type_model.pth") or os.path.join(CLOUD_TEST_DIR, "resnet_cloud_type_model.pth")
cloud_classes_json = find_file("cloud_classes.json") or os.path.join(CLOUD_TEST_DIR, "cloud_classes.json")
cloud_type_classifier = load_cloud_type_classifier(model_path=resnet_path, json_path=cloud_classes_json)
if cloud_type_classifier and cloud_type_classifier.get("is_trained"):
    print("[OK] ResNet CCSN_v2 Cloud Genus Classifier loaded and ready.", flush=True)
else:
    print("[INFO] Cloud Genus Classifier running in base mode.", flush=True)


# =============================================================================
# 5. HEADLESS BACKGROUND VIDEO STREAMER (No GUI, Low Memory)
# =============================================================================
video_worker_running = True

def headless_video_worker(stream_url, analysis_fps=1.0, use_circular_mask=True):
    """
    Consumes live RTSP or video stream entirely in memory without opening any window.
    Extracts cloud thickness, coverage, density, light, and cloud genus in real time.
    """
    global video_worker_running, latest_sky_telemetry

    # Resolve local fallback video if needed
    source = stream_url
    if isinstance(source, str) and not source.startswith("rtsp://") and not source.startswith("http://") and not source.isdigit():
        if not os.path.isabs(source):
            alt = os.path.join(PROJECT_ROOT, source)
            if os.path.exists(alt):
                source = alt

    print(f"[BACKGROUND VIDEO] Starting headless stream on: {source} (Target analysis rate: {analysis_fps:.1f} FPS)", flush=True)

    mask = None
    last_analysis_time = 0.0
    min_interval = 1.0 / max(0.1, analysis_fps)

    while video_worker_running:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[BACKGROUND VIDEO] Warning: Cannot open '{source}'. Retrying in 5 seconds...", flush=True)
            with sky_telemetry_lock:
                latest_sky_telemetry["stream_status"] = "DISCONNECTED"
                # Reset visual data while offline
                latest_sky_telemetry["cloud_thickness_pct"] = 0.0
                latest_sky_telemetry["cloud_coverage_pct"] = 0.0
                latest_sky_telemetry["cloud_density_pct"] = 0.0
                latest_sky_telemetry["num_clouds_detected"] = 0
                latest_sky_telemetry["dominant_cloud_genus"] = "Camera Offline"
                latest_sky_telemetry["dominant_cloud_code"] = "Offline"
                latest_sky_telemetry["rain_outlook"] = "Offline"
                latest_sky_telemetry["visual_meteo_score_pct"] = 0.0
            time.sleep(5)
            continue

        with sky_telemetry_lock:
            latest_sky_telemetry["stream_status"] = "ONLINE"

        frame_count = 0
        global latest_frame
        while video_worker_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # If RTSP drops, break to reconnect
                print("[BACKGROUND VIDEO] RTSP connection lost. Reconnecting...", flush=True)
                with sky_telemetry_lock:
                    latest_sky_telemetry["stream_status"] = "DISCONNECTED"
                break
            
            with latest_frame_lock:
                latest_frame = frame.copy()

            t_now = time.time()
            if (t_now - last_analysis_time) < min_interval:
                # Skip heavy computation on intermediate frames to save CPU/GPU
                continue

            last_analysis_time = t_now
            frame_count += 1
            h, w = frame.shape[:2]

            # Generate or cache ROI mask
            if mask is None or mask.shape[:2] != (h, w):
                mask, _, _ = get_roi_mask(h, w, use_circle=use_circular_mask)

            # 1. Compute Cloud Thickness & Telemetry Metrics (Mathematical Algorithm)
            try:
                metrics = calculate_thickness(frame, mask)
            except Exception as e:
                print(f"[BACKGROUND VIDEO] Error in calculate_thickness: {e}", flush=True)
                continue

            thick_val = float(metrics.get("mean_thickness", 0.0))
            cov_val = float(metrics.get("cloud_coverage", 0.0))
            dense_val = float(metrics.get("cloud_only_thickness", 0.0))
            light_val = float(metrics.get("mean_brightness", 0.0))

            # 2. YOLO Cloud Bounding Box Detection
            detections = []
            if yolo_model is not None:
                try:
                    results = yolo_model(frame, conf=0.25, verbose=False)
                    for r in results:
                        for b in r.boxes:
                            xyxy = [float(x) for x in b.xyxy[0]]
                            conf = float(b.conf[0])
                            cls_id = int(b.cls[0])
                            cls_name = yolo_model.names.get(cls_id, "cloud")
                            detections.append({
                                "box": xyxy,
                                "conf": conf,
                                "name": cls_name
                            })
                except Exception:
                    pass

            num_clouds = len(detections)

            # 3. ResNet CCSN_v2 Cloud Genus Classification (Area-Weighted Confidence Voting)
            dominant_genus = "Clear Sky"
            dom_code = "Clear"
            dom_details = get_cloud_details("Clear")

            if cov_val < 2.5:
                dominant_genus = "Clear Sky"
                dom_code = "Clear"
                dom_details = get_cloud_details("Clear")
            else:
                candidates = []
                # A. Classify YOLO boxes
                if detections and cloud_type_classifier is not None:
                    for det in detections:
                        bx1, by1 = max(0, int(det["box"][0])), max(0, int(det["box"][1]))
                        bx2, by2 = min(w - 1, int(det["box"][2])), min(h - 1, int(det["box"][3]))
                        if (bx2 - bx1) >= 16 and (by2 - by1) >= 16:
                            crop = frame[by1:by2, bx1:bx2]
                            c_code, c_name, c_conf, _ = classify_cloud_patch(cloud_type_classifier, crop)
                            area = (bx2 - bx1) * (by2 - by1)
                            candidates.append((c_code, c_name, c_conf, area))

                # B. Classify cloud mask contours if no YOLO boxes
                if not candidates and cloud_type_classifier is not None:
                    cloud_mask_u8 = (metrics["cloud_bool"] & (mask > 0)).astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(cloud_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
                        for c in cnts[:5]:
                            ca = cv2.contourArea(c)
                            if ca < 80:
                                continue
                            bx, by, bw, bh = cv2.boundingRect(c)
                            px, py = max(4, int(bw * 0.1)), max(4, int(bh * 0.1))
                            cx1, cy1 = max(0, bx - px), max(0, by - py)
                            cx2, cy2 = min(w - 1, bx + bw + px), min(h - 1, by + bh + py)
                            if (cx2 - cx1) >= 16 and (cy2 - cy1) >= 16:
                                crop = frame[cy1:cy2, cx1:cx2]
                                c_code, c_name, c_conf, _ = classify_cloud_patch(cloud_type_classifier, crop)
                                candidates.append((c_code, c_name, c_conf, ca))

                    # Sky center sample
                    ch, cw = h // 2, w // 2
                    r = min(ch, cw) // 2
                    sky_crop = frame[max(0, ch - r):min(h, ch + r), max(0, cw - r):min(w, cw + r)]
                    if sky_crop.size > 0:
                        c_code, c_name, c_conf, _ = classify_cloud_patch(cloud_type_classifier, sky_crop)
                        candidates.append((c_code, c_name, c_conf, 300.0))

                # Vote dominant genus
                if candidates:
                    votes = {}
                    code_map = {}
                    for c_code, c_name, c_conf, weight in candidates:
                        score = max(1.0, float(weight)) * c_conf
                        votes[c_name] = votes.get(c_name, 0.0) + score
                        code_map[c_name] = c_code

                    dom_name = max(votes, key=votes.get)
                    dom_code = code_map.get(dom_name, "")
                    dominant_genus = f"{dom_name} ({dom_code})" if dom_code else dom_name
                    dom_details = get_cloud_details(dom_code)

            # 4. Compute Visual Meteorological Rain Risk Score (0 - 100%)
            genus_prior = GENUS_RAIN_PRIOR.get(dom_code, 15.0)
            optical_factor = (thick_val / 100.0) * (cov_val / 100.0) * 100.0
            if cov_val < 3.0 or dom_code in ["Clear", "Ct"]:
                visual_meteo_score = 0.0
            else:
                visual_meteo_score = 0.70 * genus_prior + 0.30 * optical_factor
            visual_meteo_score = np.clip(visual_meteo_score, 0.0, 100.0)

            # 5. Atomically update thread-safe telemetry state
            with sky_telemetry_lock:
                latest_sky_telemetry = {
                    "cloud_thickness_pct": round(thick_val, 2),
                    "cloud_coverage_pct": round(cov_val, 2),
                    "cloud_density_pct": round(dense_val, 2),
                    "scene_light_pct": round(light_val, 2),
                    "num_clouds_detected": num_clouds,
                    "dominant_cloud_genus": dominant_genus,
                    "dominant_cloud_code": dom_code,
                    "atmospheric_level": dom_details.get("level", "Standard Level"),
                    "rain_outlook": dom_details.get("rain", "Fair Weather"),
                    "visual_meteo_score_pct": round(visual_meteo_score, 2),
                    "stream_status": "ONLINE (Headless RTSP)",
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

        cap.release()
        time.sleep(1.0)


# =============================================================================
# 6. CSV PERSISTENCE MANAGER
# =============================================================================
CSV_COLUMNS = [
    "timestamp",
    "temperature_c",
    "humidity_pct",
    "local_sensor_temp_c",
    "local_sensor_humidity_pct",
    "cloud_base_height_m",
    "cloud_thickness_pct",
    "cloud_coverage_pct",
    "cloud_density_pct",
    "scene_light_pct",
    "num_clouds_detected",
    "dominant_cloud_genus",
    "dominant_cloud_code",
    "atmospheric_level",
    "rain_outlook",
    "weather_main",
    "weather_description",
    "rain_1h_mm",
    "pressure_hpa",
    "wind_speed_kmh",
    "wind_direction_deg",
    "rf_rain_prediction_pct",
    "xgb_rain_prediction_pct",
    "ml_avg_rain_prediction_pct",
    "visual_meteo_score_pct",
    "final_adjusted_rain_prediction_pct",
    "predicted_rainfall_category",
    "data_source"
]

def append_to_csv(record, csv_path=DEFAULT_CSV_PATH):
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            row = {col: record.get(col, "") for col in CSV_COLUMNS}
            writer.writerow(row)
        return True
    except Exception as e:
        print(f"[ERROR] Failed writing to CSV ({csv_path}): {e}", flush=True)
        return False


# =============================================================================
# 7. DATA GATHERING & COMPREHENSIVE PREDICTION ENGINE
# =============================================================================
def fetch_local_sensor_data():
    global latest_direct_sensor
    if latest_direct_sensor is not None:
        return latest_direct_sensor

    if os.path.exists(LOCAL_SENSOR_JSON):
        try:
            with open(LOCAL_SENSOR_JSON, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "temperature_c" in data:
                    return data
        except Exception:
            pass

    try:
        r = requests.get("http://127.0.0.1:5000/api/sensor-data", timeout=0.6)
        if r.status_code == 200:
            res = r.json()
            if res.get("latest"):
                return res["latest"]
    except Exception:
        pass

    return None

def compute_final_rain_prediction(ml_avg_prob, visual_meteo_score, cloud_thickness, cloud_coverage, dominant_code):
    """
    Intelligently adjusts the final rain prediction by harmonizing:
      - ml_avg_prob (Sensor-based thermodynamic potential from RF & XGBoost)
      - visual_meteo_score (Direct camera observation of cloud genus & optical density)
    Applies atmospheric physics guardrails for dry/clear skies and severe storm clouds.
    """
    # 1. Base Weighted Blending: 55% Sensor Physics (ML), 45% Visual Genus & Thickness
    raw_blended = (0.55 * ml_avg_prob) + (0.45 * visual_meteo_score)

    # 2. Physical Guardrail A: Clear Sky / Very Thin Cloud Suppression
    # Rain cannot physically reach the ground without sufficient cloud condensation
    if cloud_coverage < 8.0 or cloud_thickness < 10.0 or dominant_code in ["Clear", "Ct"]:
        # Strongly attenuate rain probability down to near zero
        final_prob = min(raw_blended * 0.25, 8.0)

    # 3. Physical Guardrail B: Convective Storm Reinforcement (Cumulonimbus / Nimbostratus)
    # If high-threat rain cloud is physically sighted overhead with high thickness
    elif dominant_code in ["Cb", "Ns"] and cloud_thickness > 45.0:
        # Visual threat is confirmed; boost confidence
        final_prob = max(raw_blended, 0.50 * ml_avg_prob + 0.50 * visual_meteo_score)
        if ml_avg_prob > 30.0:
            final_prob = min(100.0, final_prob + 10.0)

    # 4. Standard Case
    else:
        final_prob = raw_blended

    final_prob = float(np.clip(final_prob, 0.0, 100.0))

    # Categorization
    if final_prob >= 70.0:
        category = "Heavy Rain / Storm"
    elif final_prob >= 30.0:
        category = "Light / Moderate Rain"
    else:
        category = "No Rain / Fair"

    return final_prob, category

def gather_data_cycle(lat=DEFAULT_LAT, lon=DEFAULT_LON, csv_path=DEFAULT_CSV_PATH):
    global latest_gathered_data, prev_weather_state, recent_history

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Fetch live OpenWeatherMap data
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        ow_data = resp.json()
    except Exception as e:
        print(f"[{timestamp_str}] [ERROR] OpenWeather API call failed: {e}", flush=True)
        return None

    ow_temp_c = float(ow_data['main']['temp'])
    ow_humidity_pct = float(ow_data['main']['humidity'])
    pressure_hpa = float(ow_data['main']['pressure'])

    wind_speed_ms = float(ow_data.get('wind', {}).get('speed', 0.0))
    wind_speed_kmh = wind_speed_ms * 3.6
    wind_direction_deg = float(ow_data.get('wind', {}).get('deg', 0.0))

    weather_main = ow_data.get('weather', [{}])[0].get('main', '')
    weather_desc = ow_data.get('weather', [{}])[0].get('description', '')
    rain_1h = float(ow_data.get('rain', {}).get('1h', 0.0))

    # 2. Read Local ESP32 Sensor
    sensor_info = fetch_local_sensor_data()
    local_temp = None
    local_hum = None
    if sensor_info:
        local_temp = sensor_info.get("temperature_c", sensor_info.get("temperature"))
        local_hum = sensor_info.get("humidity_pct", sensor_info.get("humidity"))

    local_pressure_hpa = sensor_info.get("pressure_hpa") if sensor_info else None
    local_wind_speed_ms = sensor_info.get("wind_speed_ms") if sensor_info else None
    local_wind_direction_deg = sensor_info.get("wind_direction_deg") if sensor_info else None
    local_rain_wetness_pct = sensor_info.get("rain_wetness_pct") if sensor_info else None
    local_rain_wet = sensor_info.get("rain_wet") if sensor_info else None

    effective_temp = float(local_temp) if local_temp is not None else ow_temp_c
    effective_hum = float(local_hum) if local_hum is not None else ow_humidity_pct
    if local_pressure_hpa is not None:
        pressure_hpa = float(local_pressure_hpa)
    if local_wind_speed_ms is not None:
        wind_speed_kmh = float(local_wind_speed_ms) * 3.6
    if local_wind_direction_deg is not None:
        wind_direction_deg = float(local_wind_direction_deg) % 360.0

    # 3. Calculate Cloud Base Height (Dew point approximation)
    dew_point_c = effective_temp - ((100.0 - effective_hum) / 5.0)
    cloud_base_height_m = max(0.0, (effective_temp - dew_point_c) * 125.0)

    # 4. Extract Real-Time Sky Telemetry from Headless Video Stream
    with sky_telemetry_lock:
        sky_data = latest_sky_telemetry.copy()

    thickness_pct = sky_data["cloud_thickness_pct"]
    coverage_pct = sky_data["cloud_coverage_pct"]
    density_pct = sky_data["cloud_density_pct"]
    scene_light_pct = sky_data["scene_light_pct"]
    num_clouds = sky_data["num_clouds_detected"]
    dom_genus = sky_data["dominant_cloud_genus"]
    dom_code = sky_data["dominant_cloud_code"]
    atmo_level = sky_data["atmospheric_level"]
    rain_outlook = sky_data["rain_outlook"]
    visual_meteo_score = sky_data["visual_meteo_score_pct"]

    # 5. 15-Minute Atmospheric Differentials for ML
    if prev_weather_state is not None:
        temp_drop_15m = effective_temp - prev_weather_state['temp_c']
        humidity_spike_15m = effective_hum - prev_weather_state['humidity_pct']
        pressure_drop_15m = pressure_hpa - prev_weather_state['pressure_hpa']
    else:
        temp_drop_15m = 0.0
        humidity_spike_15m = 0.0
        pressure_drop_15m = 0.0

    prev_weather_state = {
        'temp_c': effective_temp,
        'humidity_pct': effective_hum,
        'pressure_hpa': pressure_hpa
    }

    # 6. Wind U and V Vectors
    wind_radians = np.radians(wind_direction_deg)
    wind_u = -wind_speed_kmh * np.sin(wind_radians)
    wind_v = -wind_speed_kmh * np.cos(wind_radians)

    # 7. ML Predictions (Random Forest + XGBoost)
    feature_cols = [
        "temp_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh",
        "wind_u", "wind_v", "temp_drop_15m", "humidity_spike_15m", 
        "pressure_drop_15m"
    ]

    X_live = pd.DataFrame([{
        "temp_c": effective_temp,
        "humidity_pct": effective_hum,
        "pressure_hpa": pressure_hpa,
        "wind_speed_kmh": wind_speed_kmh,
        "wind_u": wind_u,
        "wind_v": wind_v,
        "temp_drop_15m": temp_drop_15m,
        "humidity_spike_15m": humidity_spike_15m,
        "pressure_drop_15m": pressure_drop_15m
    }], columns=feature_cols)

    rf_prob = None
    xgb_prob = None
    probs = []

    if rf_model is not None:
        try:
            rf_prob = float(rf_model.predict_proba(X_live)[:, 1][0] * 100.0)
            probs.append(rf_prob)
        except Exception:
            pass

    if xgb_model is not None:
        try:
            xgb_prob = float(xgb_model.predict_proba(X_live)[:, 1][0] * 100.0)
            probs.append(xgb_prob)
        except Exception:
            pass

    ml_avg_prob = sum(probs) / len(probs) if probs else 0.0

    # 8. Compute Adjusted Final Rain Prediction (Harmonizing ML + Camera Meteorological Outlook)
    final_prob, rainfall_category = compute_final_rain_prediction(
        ml_avg_prob=ml_avg_prob,
        visual_meteo_score=visual_meteo_score,
        cloud_thickness=thickness_pct,
        cloud_coverage=coverage_pct,
        dominant_code=dom_code
    )

    # 9. Assemble Complete Record
    record = {
        "timestamp": timestamp_str,
        "temperature_c": round(effective_temp, 2),
        "humidity_pct": round(effective_hum, 2),
        "local_sensor_temp_c": round(float(local_temp), 2) if local_temp is not None else "",
        "local_sensor_humidity_pct": round(float(local_hum), 2) if local_hum is not None else "",
        "local_sensor_pressure_hpa": round(float(local_pressure_hpa), 2) if local_pressure_hpa is not None else "",
        "local_sensor_wind_speed_ms": round(float(local_wind_speed_ms), 2) if local_wind_speed_ms is not None else "",
        "local_sensor_wind_direction_deg": round(float(local_wind_direction_deg), 1) if local_wind_direction_deg is not None else "",
        "wind_direction_cardinal": sensor_info.get("wind_direction_cardinal", "") if sensor_info else "",
        "rain_sensor_wetness_pct": round(float(local_rain_wetness_pct), 1) if local_rain_wetness_pct is not None else "",
        "rain_sensor_wet": local_rain_wet if local_rain_wet is not None else "",
        "cloud_base_height_m": round(cloud_base_height_m, 2),
        "cloud_thickness_pct": round(thickness_pct, 2),
        "cloud_coverage_pct": round(coverage_pct, 2),
        "cloud_density_pct": round(density_pct, 2),
        "scene_light_pct": round(scene_light_pct, 2),
        "num_clouds_detected": num_clouds,
        "dominant_cloud_genus": dom_genus,
        "dominant_cloud_code": dom_code,
        "atmospheric_level": atmo_level,
        "rain_outlook": rain_outlook,
        "weather_main": weather_main,
        "weather_description": weather_desc,
        "rain_1h_mm": round(rain_1h, 2),
        "pressure_hpa": round(pressure_hpa, 2),
        "wind_speed_kmh": round(wind_speed_kmh, 2),
        "wind_direction_deg": round(wind_direction_deg, 2),
        "rf_rain_prediction_pct": round(rf_prob, 2) if rf_prob is not None else "",
        "xgb_rain_prediction_pct": round(xgb_prob, 2) if xgb_prob is not None else "",
        "ml_avg_rain_prediction_pct": round(ml_avg_prob, 2),
        "visual_meteo_score_pct": round(visual_meteo_score, 2),
        "final_adjusted_rain_prediction_pct": round(final_prob, 2),
        "predicted_rainfall_category": rainfall_category,
        "data_source": "OpenWeather+HeadlessRTSP+" + ("ESP32" if any(value is not None for value in (local_temp, local_hum, local_pressure_hpa, local_wind_speed_ms, local_wind_direction_deg)) else "Ambient")
    }

    # 10. Persist to CSV
    append_to_csv(record, csv_path=csv_path)

    # 11. Memory Update
    latest_gathered_data = record
    recent_history.append(record)
    if len(recent_history) > 200:
        recent_history.pop(0)

    # 12. Console Logging
    print(f"\n==================================================================", flush=True)
    print(f"[TELEMETRY GATHERED & LOGGED TO CSV] {timestamp_str}", flush=True)
    print(f"  * Temperature         : {record['temperature_c']} C " + (f"(ESP32 SHT45)" if local_temp is not None else "(Ambient)"), flush=True)
    print(f"  * Humidity            : {record['humidity_pct']} %RH", flush=True)
    print(f"  * Cloud Base Height   : {record['cloud_base_height_m']} m", flush=True)
    print(f"  * Sky Thickness       : {record['cloud_thickness_pct']}% | Coverage: {record['cloud_coverage_pct']}%", flush=True)
    print(f"  * Dominant Genus      : {record['dominant_cloud_genus']} -> {record['atmospheric_level']}", flush=True)
    print(f"  * Genus Rain Outlook  : {record['rain_outlook']}", flush=True)
    print(f"  * ML Sensor Ensemble  : RF: {record['rf_rain_prediction_pct']}% | XGB: {record['xgb_rain_prediction_pct']}% -> Avg: {record['ml_avg_rain_prediction_pct']}%", flush=True)
    print(f"  * Visual Meteo Risk   : {record['visual_meteo_score_pct']}%", flush=True)
    print(f"  * FINAL RAIN CHANCE   : {record['final_adjusted_rain_prediction_pct']}% -> [{record['predicted_rainfall_category']}]", flush=True)
    print(f"  * Appended to CSV     : {csv_path}", flush=True)
    print(f"==================================================================\n", flush=True)

    return record


# =============================================================================
# 8. PERIODIC DATA GATHERING WORKER
# =============================================================================
collector_running = True

def background_gathering_worker(interval_sec, lat, lon, csv_path):
    global collector_running
    print(f"[INFO] Periodic CSV logging started (Every {interval_sec}s -> {csv_path})", flush=True)
    while collector_running:
        try:
            gather_data_cycle(lat=lat, lon=lon, csv_path=csv_path)
        except Exception as e:
            print(f"[ERROR] Periodic gathering cycle error: {e}", flush=True)
        time.sleep(interval_sec)


# =============================================================================
# 9. FLASK REST API & WEB DASHBOARD
# =============================================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cloud Chaser - Headless Data Gathering Server</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0e14;
            --card-bg: #141923;
            --border: #232c3d;
            --accent-cyan: #38bdf8;
            --accent-blue: #60a5fa;
            --accent-green: #4ade80;
            --accent-amber: #fbbf24;
            --accent-red: #f87171;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 24px;
        }
        .container { max-width: 1300px; margin: 0 auto; }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        h1 { font-size: 22px; font-weight: 800; color: var(--accent-cyan); display: flex; align-items: center; gap: 8px; }
        .stream-badge {
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            padding: 3px 10px;
            border-radius: 12px;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
            border: 1px solid rgba(56, 189, 248, 0.4);
            font-weight: 600;
        }
        .btn-group { display: flex; gap: 10px; }
        .btn {
            background: #1e2638;
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            transition: all 0.2s;
        }
        .btn:hover { background: #2b364e; border-color: #64748b; }
        .btn-primary { background: #0284c7; border-color: #0369a1; }
        .btn-primary:hover { background: #0369a1; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }
        .card-label {
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--text-sub);
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .card-value {
            font-size: 24px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
        }
        .card-sub {
            font-size: 12px;
            color: var(--text-sub);
            margin-top: 6px;
        }
        .highlight-card {
            border: 1px solid rgba(251, 191, 36, 0.5);
            background: rgba(251, 191, 36, 0.05);
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 700;
        }
        .badge-heavy { background: rgba(248, 113, 113, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }
        .badge-light { background: rgba(251, 191, 36, 0.2); color: var(--accent-amber); border: 1px solid var(--accent-amber); }
        .badge-none { background: rgba(74, 222, 128, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .table-container {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        .table-header {
            padding: 12px 16px;
            background: #18202f;
            border-bottom: 1px solid var(--border);
            font-weight: 700;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
        }
        th, td {
            padding: 9px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th {
            background: #141923;
            color: var(--text-sub);
            font-weight: 600;
            text-transform: uppercase;
        }
        tr:hover { background-color: #1e2638; }
    </style>
</head>
<body data-sensor-timestamp="{{ direct_sensor.timestamp if direct_sensor else '' }}">
    <div class="container">
        <header>
            <div>
                <h1>CLOUD CHASER - HEADLESS DATA GATHERING SERVER</h1>
                <p style="color: var(--text-sub); font-size: 13px; margin-top: 4px;">
                    Continuous Background Stream Ingestion (No Display) &bull; ResNet Genus AI &bull; Dual-Harmonized Rain Predictions
                </p>
            </div>
            <div class="btn-group">
                <span class="stream-badge">{{ telemetry.stream_status }}</span>
                <button class="btn btn-primary" onclick="triggerGather()">Gather Now</button>
                <a href="/api/csv" class="btn">Download CSV</a>
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <div class="card-label">Temperature</div>
                <div class="card-value">{{ direct_sensor.temperature_c if direct_sensor else (latest.temperature_c if latest else '--') }} °C</div>
                <div class="card-sub">{{ 'ESP32 SHT45' if direct_sensor else 'OpenWeather' }}</div>
            </div>
            <div class="card">
                <div class="card-label">Humidity</div>
                <div class="card-value">{{ direct_sensor.humidity_pct if direct_sensor else (latest.humidity_pct if latest else '--') }} %</div>
                <div class="card-sub">Relative Humidity (%RH)</div>
            </div>
            <div class="card">
                <div class="card-label">Barometric Pressure</div>
                <div class="card-value">{{ direct_sensor.pressure_hpa if direct_sensor and direct_sensor.pressure_hpa is defined else (latest.pressure_hpa if latest else '--') }} hPa</div>
                <div class="card-sub">{{ 'ESP32 BMP580' if direct_sensor and direct_sensor.pressure_hpa is defined else 'OpenWeather' }}</div>
            </div>
            <div class="card">
                <div class="card-label">Wind Speed</div>
                <div class="card-value">{{ (direct_sensor.wind_speed_ms * 3.6)|round(2) if direct_sensor and direct_sensor.wind_speed_ms is defined else (latest.wind_speed_kmh if latest else '--') }} km/h</div>
                <div class="card-sub">{{ direct_sensor.wind_speed_ms if direct_sensor and direct_sensor.wind_speed_ms is defined else '--' }} m/s from ESP32</div>
            </div>
            <div class="card">
                <div class="card-label">Wind Direction</div>
                <div class="card-value">{{ direct_sensor.wind_direction_deg if direct_sensor and direct_sensor.wind_direction_deg is defined else (latest.wind_direction_deg if latest else '--') }}°</div>
                <div class="card-sub">{{ direct_sensor.wind_direction_cardinal if direct_sensor and direct_sensor.wind_direction_cardinal is defined else (latest.wind_direction_cardinal if latest and latest.wind_direction_cardinal else 'OpenWeather / unavailable') }}</div>
            </div>
            <div class="card">
                <div class="card-label">Rain Sensor</div>
                <div class="card-value">{{ direct_sensor.rain_wetness_pct if direct_sensor and direct_sensor.rain_wetness_pct is defined else '--' }} % wetness</div>
                <div class="card-sub">{% if direct_sensor and direct_sensor.rain_wet is defined %}{{ 'Wet' if direct_sensor.rain_wet else 'Dry' }}{% else %}No ESP32 reading{% endif %}</div>
            </div>
            <div class="card">
                <div class="card-label">Cloud Base Height</div>
                <div class="card-value" style="color: var(--accent-cyan);">{{ latest.cloud_base_height_m if latest else '--' }} m</div>
                <div class="card-sub">Lifting Condensation Level</div>
            </div>
            <div class="card">
                <div class="card-label">Sky Telemetry (Headless)</div>
                <div class="card-value" style="color: var(--accent-blue);">{{ latest.cloud_thickness_pct if latest else '--' }} %</div>
                <div class="card-sub">Cover: {{ latest.cloud_coverage_pct if latest else '--' }}% | Density: {{ latest.cloud_density_pct if latest else '--' }}%</div>
            </div>
            <div class="card">
                <div class="card-label">Dominant Cloud Genus</div>
                <div class="card-value" style="font-size: 18px; color: #e2e8f0;">{{ latest.dominant_cloud_genus if latest else '--' }}</div>
                <div class="card-sub">{{ latest.atmospheric_level if latest else '--' }}</div>
            </div>
            <div class="card">
                <div class="card-label">Meteorological Outlook</div>
                <div class="card-value" style="font-size: 16px; color: var(--accent-amber);">{{ latest.rain_outlook if latest else '--' }}</div>
                <div class="card-sub">Visual Risk Score: {{ latest.visual_meteo_score_pct if latest else '--' }}%</div>
            </div>
            <div class="card">
                <div class="card-label">ML Sensor Ensemble (15m)</div>
                <div class="card-value">{{ latest.ml_avg_rain_prediction_pct if latest else '--' }} %</div>
                <div class="card-sub">RF: {{ latest.rf_rain_prediction_pct if latest else '--' }}% | XGB: {{ latest.xgb_rain_prediction_pct if latest else '--' }}%</div>
            </div>
            <div class="card highlight-card">
                <div class="card-label" style="color: var(--accent-amber);">FINAL ADJUSTED RAIN CHANCE</div>
                <div class="card-value" style="color: var(--accent-amber); font-size: 26px;">{{ latest.final_adjusted_rain_prediction_pct if latest else '--' }} %</div>
                <div class="card-sub">
                    {% if latest %}
                        {% if latest.final_adjusted_rain_prediction_pct >= 70 %}
                            <span class="badge badge-heavy">HEAVY RAIN / STORM</span>
                        {% elif latest.final_adjusted_rain_prediction_pct >= 30 %}
                            <span class="badge badge-light">LIGHT / MODERATE RAIN</span>
                        {% else %}
                            <span class="badge badge-none">NO RAIN / FAIR</span>
                        {% endif %}
                    {% endif %}
                    &bull; {{ latest.timestamp if latest else '' }}
                </div>
            </div>
        </div>

        <div class="table-container">
            <div class="table-header">
                <span>Recent CSV Telemetry Logs</span>
                <span style="color: var(--text-sub); font-weight: normal; font-size: 12px;">Auto-refreshes every 10s</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Temp (°C)</th>
                        <th>Humidity (%)</th>
                        <th>Pressure (hPa)</th>
                        <th>Wind (km/h)</th>
                        <th>Direction</th>
                        <th>Cloud Base</th>
                        <th>Thickness</th>
                        <th>Coverage</th>
                        <th>Dominant Genus</th>
                        <th>ML Rain %</th>
                        <th>Visual Risk %</th>
                        <th>FINAL RAIN %</th>
                        <th>Category</th>
                        <th>Rain Sensor</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in history[-15:]|reverse %}
                    <tr>
                        <td>{{ row.timestamp }}</td>
                        <td>{{ row.temperature_c }}</td>
                        <td>{{ row.humidity_pct }}</td>
                        <td>{{ row.pressure_hpa }}</td>
                        <td>{{ row.wind_speed_kmh }}</td>
                        <td>{{ row.wind_direction_deg }}° {{ row.wind_direction_cardinal }}</td>
                        <td>{{ row.cloud_base_height_m }} m</td>
                        <td>{{ row.cloud_thickness_pct }}%</td>
                        <td>{{ row.cloud_coverage_pct }}%</td>
                        <td style="color: var(--accent-cyan);">{{ row.dominant_cloud_genus }}</td>
                        <td>{{ row.ml_avg_rain_prediction_pct }}%</td>
                        <td>{{ row.visual_meteo_score_pct }}%</td>
                        <td style="font-weight: 700; color: var(--accent-amber);">{{ row.final_adjusted_rain_prediction_pct }}%</td>
                        <td>{{ row.predicted_rainfall_category }}</td>
                        <td>{% if row.rain_sensor_wet != '' %}{{ row.rain_sensor_wetness_pct }}% / {{ 'Wet' if row.rain_sensor_wet else 'Dry' }}{% else %}—{% endif %}</td>
                    </tr>
                    {% else %}
                    <tr><td colspan="15" style="text-align: center; color: var(--text-sub);">No records gathered yet. Click "Gather Now" above.</td></tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        async function triggerGather() {
            try {
                const res = await fetch('/api/gather');
                const data = await res.json();
                if (data.status === 'success') location.reload();
            } catch (err) { alert('Gather error: ' + err); }
        }
        setInterval(() => {
            fetch('/api/latest')
                .then(r => r.json())
                .then(data => {
                    if (data && data.latest) {
                        const cur = document.body.innerHTML;
                        if (!cur.includes(data.latest.timestamp)) location.reload();
                    }
                    if (data && data.direct_sensor && data.direct_sensor.timestamp !== document.body.dataset.sensorTimestamp) location.reload();
                }).catch(() => {});
        }, 10000);
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index_dashboard():
    with sky_telemetry_lock:
        current_telemetry = latest_sky_telemetry.copy()
    return render_template_string(
        DASHBOARD_HTML,
        latest=latest_gathered_data,
        direct_sensor=latest_direct_sensor,
        telemetry=current_telemetry,
        history=recent_history
    )

@app.route('/api/gather', methods=['GET', 'POST'])
def api_gather():
    lat = request.args.get('lat', default=DEFAULT_LAT, type=float)
    lon = request.args.get('lon', default=DEFAULT_LON, type=float)
    record = gather_data_cycle(lat=lat, lon=lon)
    if record:
        return jsonify({"status": "success", "data": record}), 200
    return jsonify({"status": "error", "message": "Failed gathering data"}), 500

@app.route('/api/latest', methods=['GET'])
def api_latest():
    with sky_telemetry_lock:
        current_telemetry = latest_sky_telemetry.copy()
    return jsonify({
        "status": "success",
        "latest": latest_gathered_data,
        "direct_sensor": latest_direct_sensor,
        "sky_telemetry": current_telemetry,
        "csv_path": DEFAULT_CSV_PATH
    })

@app.route('/api/weather', methods=['GET'])
def api_weather():
    if not latest_gathered_data:
        return jsonify({"status": "error", "message": "No data gathered yet"}), 503
    
    with sky_telemetry_lock:
        stream_status = latest_sky_telemetry.get("stream_status", "DISCONNECTED")
    
    response_data = {
        "temperature_c": latest_gathered_data.get("temperature_c", 0.0),
        "humidity_pct": latest_gathered_data.get("humidity_pct", 0.0),
        "pressure_hpa": latest_gathered_data.get("pressure_hpa", 0.0),
        "wind_speed_kmh": latest_gathered_data.get("wind_speed_kmh", 0.0),
        "wind_direction_deg": latest_gathered_data.get("wind_direction_deg", 0.0),
        "wind_direction_cardinal": latest_gathered_data.get("wind_direction_cardinal", ""),
        "rain_sensor_wetness_pct": latest_gathered_data.get("rain_sensor_wetness_pct", ""),
        "rain_sensor_wet": latest_gathered_data.get("rain_sensor_wet", ""),
        "cloud_base_height_m": latest_gathered_data.get("cloud_base_height_m", 0.0),
        "cloud_thickness_pct": latest_gathered_data.get("cloud_thickness_pct", 0.0),
        "rainfall_probability_pct": latest_gathered_data.get("final_adjusted_rain_prediction_pct", 0.0),
        "dominant_cloud_genus": latest_gathered_data.get("dominant_cloud_genus", "Clear Sky"),
        "predicted_rainfall_category": latest_gathered_data.get("predicted_rainfall_category", "No Rain / Fair"),
        "stream_status": stream_status
    }
    return jsonify(response_data)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with latest_frame_lock:
                if latest_frame is None:
                    frame = None
                else:
                    frame = latest_frame.copy()
            
            if frame is None:
                time.sleep(0.1)
                continue
                
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ret:
                time.sleep(0.1)
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
            
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry', methods=['GET'])
def api_telemetry():
    with sky_telemetry_lock:
        current_telemetry = latest_sky_telemetry.copy()
    return jsonify({
        "status": "success",
        "telemetry": current_telemetry
    })

@app.route('/api/history', methods=['GET'])
def api_history():
    limit = request.args.get('limit', default=50, type=int)
    return jsonify({
        "status": "success",
        "count": len(recent_history),
        "history": recent_history[-limit:]
    })

@app.route('/api/csv', methods=['GET'])
def api_download_csv():
    if os.path.exists(DEFAULT_CSV_PATH):
        return send_file(
            DEFAULT_CSV_PATH,
            mimetype="text/csv",
            as_attachment=True,
            download_name="gathered_weather_data.csv"
        )
    return jsonify({"status": "error", "message": "CSV file does not exist yet"}), 404

@app.route('/api/sensor-data', methods=['POST', 'GET'])
@app.route('/api/sensor', methods=['POST', 'GET'])
def api_sensor_endpoint():
    global latest_direct_sensor
    if request.method == 'POST':
        payload = request.get_json(silent=True) or request.form.to_dict()
        if not payload:
            try:
                payload = json.loads(request.data.decode('utf-8'))
            except Exception:
                pass

        if payload:
            temp = payload.get('temperature', payload.get('temp_c', payload.get('temp')))
            hum = payload.get('humidity', payload.get('humidity_pct', payload.get('humid')))
            if temp is not None and hum is not None:
                try:
                    latest_direct_sensor = {
                        "temperature_c": float(temp),
                        "humidity_pct": float(hum),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    numeric_fields = {
                        "pressure_hpa": ("pressure_hpa", "pressure"),
                        "wind_speed_ms": ("wind_speed_ms", "wind_speed_mps"),
                        "wind_direction_deg": ("wind_direction_deg", "wind_direction"),
                        "rain_wetness_pct": ("rain_wetness_pct", "rain_percent")
                    }
                    for output_key, input_keys in numeric_fields.items():
                        raw_value = next((payload[key] for key in input_keys if payload.get(key) not in (None, "")), None)
                        if raw_value is not None:
                            value = float(raw_value)
                            if not np.isfinite(value):
                                raise ValueError(f"{output_key} must be a finite number")
                            latest_direct_sensor[output_key] = value
                    if "wind_speed_ms" in latest_direct_sensor and latest_direct_sensor["wind_speed_ms"] < 0:
                        raise ValueError("wind_speed_ms cannot be negative")
                    if "wind_direction_deg" in latest_direct_sensor:
                        latest_direct_sensor["wind_direction_deg"] %= 360.0
                    if "wind_direction_cardinal" in payload:
                        latest_direct_sensor["wind_direction_cardinal"] = str(payload["wind_direction_cardinal"])
                    if "rain_wet" in payload:
                        raw_wet = payload["rain_wet"]
                        latest_direct_sensor["rain_wet"] = raw_wet if isinstance(raw_wet, bool) else str(raw_wet).lower() in ("true", "1", "yes", "wet")
                    record = gather_data_cycle()
                    return jsonify({"status": "success", "record": record}), 200
                except (ValueError, TypeError) as e:
                    return jsonify({"status": "error", "message": str(e)}), 400

        return jsonify({"status": "error", "message": "Temperature and humidity are required"}), 400

    return jsonify({"status": "success", "sensor": latest_direct_sensor})

@app.route('/health', methods=['GET'])
def api_health():
    with sky_telemetry_lock:
        status_str = latest_sky_telemetry.get("stream_status", "unknown")
    return jsonify({
        "status": "healthy",
        "rf_model_loaded": rf_model is not None,
        "xgb_model_loaded": xgb_model is not None,
        "yolo_model_loaded": yolo_model is not None,
        "cloud_type_loaded": cloud_type_classifier is not None,
        "stream_status": status_str,
        "latest_timestamp": latest_gathered_data.get('timestamp') if latest_gathered_data else None
    })


# =============================================================================
# 10. ENTRY POINT & CLI RUNNER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Cloud Chaser - Headless Data Gathering Server")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on (default: 5001)")
    parser.add_argument("--video", type=str, default=DEFAULT_RTSP_URL,
                        help=f"RTSP stream URL or local video file (default: {DEFAULT_RTSP_URL})")
    parser.add_argument("--video-fps", type=float, default=1.0,
                        help="Headless video analysis rate in FPS (default: 1.0 FPS for minimal memory/CPU)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Periodic CSV logging interval in seconds (default: 10s; 0 to disable periodic)")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Latitude (default: 12.3715)")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Longitude (default: 123.6247)")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Output CSV path")
    args = parser.parse_args()

    # 1. Start Headless Background Video Streamer (No GUI, Low Memory)
    video_thread = threading.Thread(
        target=headless_video_worker,
        args=(args.video, args.video_fps, True),
        daemon=True
    )
    video_thread.start()

    # Give video thread a moment to initialize
    time.sleep(1.0)

    # 2. Run initial gathering cycle
    print("\n[STARTUP] Running initial telemetry gathering cycle...", flush=True)
    gather_data_cycle(lat=args.lat, lon=args.lon, csv_path=args.csv)

    # 3. Start periodic background CSV logging worker
    if args.interval > 0:
        gather_thread = threading.Thread(
            target=background_gathering_worker,
            args=(args.interval, args.lat, args.lon, args.csv),
            daemon=True
        )
        gather_thread.start()

    print(f"\n==================================================================", flush=True)
    print(f"  Cloud Chaser Data Gathering Server running on: http://0.0.0.0:{args.port}", flush=True)
    print(f"  Web Dashboard   : http://localhost:{args.port}", flush=True)
    print(f"  Video Source    : {args.video} (HEADLESS / NO DISPLAY)", flush=True)
    print(f"  Video Rate      : {args.video_fps} FPS analysis", flush=True)
    print(f"  CSV Output      : {args.csv}", flush=True)
    print(f"  Logging Interval: Every {args.interval} seconds", flush=True)
    print(f"==================================================================\n", flush=True)

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
