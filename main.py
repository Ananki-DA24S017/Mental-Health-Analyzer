from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import joblib
import numpy as np
import os
import logging
import time
import psutil
import threading
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest
)
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mental_health_api")

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

app = FastAPI(title="Mental Health Analyzer API")

# --- Categories ---
categories = ["Normal", "Depression", "Anxiety", "Bipolar", "Personality disorder", "Stress", "Suicidal"]

# --- Request/Response Models ---
class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    category: str
    confidence: float

# --- App Metrics ---
API_REQUEST_COUNT = Counter('mental_health_api_requests_total', 'Total API requests', ['endpoint', 'method'])
API_CALLS_TOTAL = Counter('mental_health_api_calls_total', 'Total number of API calls')
PREDICTION_LATENCY = Histogram('mental_health_prediction_latency_seconds', 'Time for prediction processing')
CATEGORY_COUNTS = Gauge('mental_health_category_count', 'Count of predictions by category', ['category'])
CATEGORY_DISTRIBUTION = Gauge('mental_health_category_distribution_percent', 'Distribution of mental health categories', ['category'])

for cat in categories:
    CATEGORY_COUNTS.labels(category=cat).set(0)

# --- System Metrics ---
CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU Utilization percentage')
MEMORY_USAGE_BYTES = Gauge('system_memory_usage_bytes', 'Memory Usage in bytes')
DISK_FREE_PERCENT = Gauge('system_disk_free_percent', 'Disk Free Space percentage')
DISK_READ_RATE = Gauge('system_disk_read_bytes_per_sec', 'Disk read rate in bytes per second')
DISK_WRITE_RATE = Gauge('system_disk_write_bytes_per_sec', 'Disk write rate in bytes per second')
NET_IO_SENT = Gauge('system_network_io_sent_bytes_per_sec', 'Network IO sent in bytes per second')
NET_IO_RECEIVED = Gauge('system_network_io_received_bytes_per_sec', 'Network IO received in bytes per second')

prev_disk_io = psutil.disk_io_counters()
prev_net_io = psutil.net_io_counters()
prev_time = time.time()

# --- Helper Functions ---
def update_prediction_metrics(category):
    if category in categories:
        current = CATEGORY_COUNTS.labels(category=category)._value.get()
        CATEGORY_COUNTS.labels(category=category).set(current + 1)
        API_CALLS_TOTAL.inc()
    else:
        logger.warning(f"Unknown category: {category}")

def update_distribution():
    total = sum(CATEGORY_COUNTS.labels(category=cat)._value.get() for cat in categories)
    if total > 0:
        for cat in categories:
            count = CATEGORY_COUNTS.labels(category=cat)._value.get()
            percent = (count / total) * 100
            CATEGORY_DISTRIBUTION.labels(category=cat).set(percent)

def update_system_metrics():
    global prev_disk_io, prev_net_io, prev_time
    try:
        # CPU and Memory
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE_BYTES.set(psutil.virtual_memory().used)

        # Disk Space
        disk = psutil.disk_usage('/')
        DISK_FREE_PERCENT.set(100 - disk.percent)

        # Disk and Network I/O rates
        current_time = time.time()
        time_diff = current_time - prev_time
        if time_diff > 0:
            current_disk_io = psutil.disk_io_counters()
            current_net_io = psutil.net_io_counters()

            # Disk rates
            DISK_READ_RATE.set((current_disk_io.read_bytes - prev_disk_io.read_bytes) / time_diff)
            DISK_WRITE_RATE.set((current_disk_io.write_bytes - prev_disk_io.write_bytes) / time_diff)

            # Network rates
            NET_IO_SENT.set((current_net_io.bytes_sent - prev_net_io.bytes_sent) / time_diff)
            NET_IO_RECEIVED.set((current_net_io.bytes_recv - prev_net_io.bytes_recv) / time_diff)

            # Update previous snapshots
            prev_disk_io = current_disk_io
            prev_net_io = current_net_io
            prev_time = current_time

    except Exception as e:
        logger.error(f"System metrics error: {str(e)}")

def metrics_updater():
    while True:
        try:
            update_distribution()
            update_system_metrics()
            time.sleep(15)
        except Exception as e:
            logger.error(f"Metrics update loop error: {str(e)}")
            time.sleep(60)

# Start background thread
threading.Thread(target=metrics_updater, daemon=True).start()

# --- API Routes ---
@app.get("/")
async def root():
    API_REQUEST_COUNT.labels(endpoint="/", method="GET").inc()
    API_CALLS_TOTAL.inc()
    return {"message": "Welcome to Mental Health Analyzer API"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TextInput):
    API_REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()
    API_CALLS_TOTAL.inc()
    try:
        logger.info(f"Prediction input: {input_data.text[:50]}...")
        with PREDICTION_LATENCY.time():
            prediction = model.predict([input_data.text])[0]
            try:
                proba = model.predict_proba([input_data.text])[0]
                confidence = float(max(proba))
            except:
                confidence = 1.0

        # Determine category label
        if isinstance(prediction, (int, float, np.integer)):
            category = categories[int(prediction)]
        elif isinstance(prediction, str):
            category = prediction
        else:
            category = str(prediction)

        update_prediction_metrics(category)
        logger.info(f"Predicted: {category} ({confidence:.2f})")
        return PredictionOutput(category=category, confidence=confidence)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction error.")

@app.get("/metrics")
async def metrics():
    API_REQUEST_COUNT.labels(endpoint="/metrics", method="GET").inc()
    API_CALLS_TOTAL.inc()
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    API_REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    API_CALLS_TOTAL.inc()
    return {"status": "healthy"}
