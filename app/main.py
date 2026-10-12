import os
import json
import joblib
import psycopg2
import redis

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="ML Prediction API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

model = joblib.load("model.pkl")

FLOWER_NAMES = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}

redis_port_raw = os.getenv("REDIS_PORT", "6379")

if "tcp://" in str(redis_port_raw):
    redis_port = int(str(redis_port_raw).split(":")[-1])
else:
    redis_port = int(redis_port_raw)

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=redis_port,
    decode_responses=True
)

class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        database=os.getenv("POSTGRES_DB", "ml_db"),
        user=os.getenv("POSTGRES_USER", "ml_user"),
        password=os.getenv("POSTGRES_PASSWORD", "ml_password")
    )

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as file:
        return file.read()

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(data: PredictionInput):
    cache_key = (
        f"{data.sepal_length}:"
        f"{data.sepal_width}:"
        f"{data.petal_length}:"
        f"{data.petal_width}"
    )

    cached_prediction = redis_client.get(cache_key)

    if cached_prediction:
        prediction_code = int(cached_prediction)
        return {
            "prediction": prediction_code,
            "species_name": FLOWER_NAMES.get(prediction_code, "Unknown"),
            "source": "redis_cache"
        }

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (sepal_length, sepal_width, petal_length, petal_width, prediction)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (data.sepal_length, data.sepal_width, data.petal_length, data.petal_width, prediction)
    )

    conn.commit()
    cursor.close()
    conn.close()

    redis_client.set(cache_key, prediction, ex=3600)

    return {
        "prediction": prediction,
        "species_name": FLOWER_NAMES.get(prediction, "Unknown"),
        "source": "ml_model"
    }
