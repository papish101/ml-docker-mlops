import os
import json
import joblib
import psycopg2
import redis

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="ML Prediction API"
)

Instrumentator().instrument(app).expose(app)
# Load ML model
model = joblib.load("model.pkl")


# Redis connection
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
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


@app.get("/")
def home():

    return {
        "message": "ML Prediction API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    # Create cache key
    cache_key = (
        f"{data.sepal_length}:"
        f"{data.sepal_width}:"
        f"{data.petal_length}:"
        f"{data.petal_width}"
    )

    # Check Redis cache
    cached_prediction = redis_client.get(cache_key)

    if cached_prediction:

        return {
            "prediction": int(cached_prediction),
            "source": "redis_cache"
        }

    # ML prediction
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = int(model.predict(features)[0])

    # Save to PostgreSQL
    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
            prediction
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
            prediction
        )
    )

    conn.commit()

    cursor.close()
    conn.close()

    # Save to Redis
    redis_client.set(
        cache_key,
        prediction,
        ex=3600
    )

    return {
        "prediction": prediction,
        "source": "ml_model"
    }
