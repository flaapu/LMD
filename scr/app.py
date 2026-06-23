import os
import sys
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

# Asegurar que Python reconozca la raíz del proyecto para las rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(
    title="AndesLink Churn Prediction API",
    description="API local para la predicción de abandono de clientes bajo prácticas MLOps.",
    version="1.0.0"
)

# Cargar el modelo entrenado y serializado de forma segura
MODEL_PATH = "models/best_model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# DEFINICIÓN DEL CONTRATO DE ENTRADA (Requisito obligatorio de la consigna)
# Forzamos a que la entrada sea una lista de exactamente 22 números flotantes


class PredictionInput(BaseModel):
    features: conlist(float, min_length=22, max_length=22)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "API de Predicción de Churn para AndesLink funcionando correctamente."
    }


@app.post("/predict")
def predict_churn(data: PredictionInput):
    # Si el modelo no se encuentra en el contenedor
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo predictivo no está disponible. Verifique el pipeline de entrenamiento."
        )

    try:
        # data.features ya viene validado por Pydantic con longitud 22
        prediction = model.predict([data.features])
        probability = model.predict_proba([data.features])[0][1]

        # DEFINICIÓN DEL CONTRATO DE SALIDA CONSISTENTE
        return {
            "status": "success",
            "churn_prediction": int(prediction[0]),
            "churn_probability": round(float(probability), 4)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar la inferencia: {str(e)}"
        )
