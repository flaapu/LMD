from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import numpy as np

app = FastAPI(title="API de Predicción de Churn", version="2.0")

# Definir la ruta del modelo
MODEL_PATH = os.path.join("models", "best_model.joblib")


@app.get("/")
def home():
    return {"message": "API de Predicción de Churn funcionando correctamente."}


@app.post("/predict")
def predict(features: list[float]):
    """
    Recibe una lista de números (features) y devuelve la predicción.
    Ejemplo de entrada: [1.0, 24.5, 0.0, 120.0]
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=404, detail="El modelo entrenado no existe. Ejecute el entrenamiento primero.")

    try:
        # Cargar el modelo
        model = joblib.load(MODEL_PATH)

        # Convertir datos de entrada a formato correcto para sklearn
        data = np.array(features).reshape(1, -1)

        # Predecir
        prediction = int(model.predict(data)[0])
        probability = float(model.predict_proba(data)[0][1])

        return {
            "churn_prediction": prediction,
            "churn_probability": probability,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error en la predicción: {str(e)}")
