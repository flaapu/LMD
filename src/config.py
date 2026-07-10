import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
PREPROCESSOR_PATH = os.getenv(
    "PREPROCESSOR_PATH", "models/preprocessor.joblib")
API_PORT = int(os.getenv("API_PORT", "8080"))
API_URL = os.getenv("API_URL", "http://localhost:8080/predict")
