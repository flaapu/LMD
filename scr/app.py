"""
app.py – API de inferencia de churn (AndesLink Servicios Digitales S.A.)

Recibe datos crudos del cliente, aplica el pipeline de preprocesamiento
y devuelve la predicción junto con su probabilidad.
"""

import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join("models", "best_model.joblib")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.joblib")

artifacts: dict = {}


def _load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Modelo no encontrado en '{MODEL_PATH}'. "
            "Ejecute primero: python scr/data.py && python scr/train.py"
        )
    if not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(
            f"Preprocesador no encontrado en '{PREPROCESSOR_PATH}'. "
            "Ejecute primero: python scr/data.py"
        )
    artifacts["model"] = joblib.load(MODEL_PATH)
    artifacts["preprocessor"] = joblib.load(PREPROCESSOR_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield
    artifacts.clear()


app = FastAPI(
    title="API de Predicción de Churn — AndesLink",
    description="Predice la probabilidad de abandono de un cliente a partir de sus datos de servicio.",
    version="2.0",
    lifespan=lifespan,
)


# ── Schema de entrada: datos crudos del cliente ──────────────

class ClienteInput(BaseModel):
    tenure_months: int = Field(..., ge=0)
    monthly_charge: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    late_payments: int = Field(..., ge=0)
    avg_monthly_usage_gb: float = Field(..., ge=0)
    num_products: int = Field(..., ge=1)
    customer_age: int = Field(..., ge=18, le=100)
    has_streaming: int = Field(..., ge=0, le=1)
    has_security_pack: int = Field(..., ge=0, le=1)
    is_promo: int = Field(..., ge=0, le=1)
    contract_type: str = Field(..., description="mensual | anual | bianual")
    payment_method: str = Field(..., description="transferencia | debito | efectivo | credito")
    internet_service: str = Field(..., description="cable | fibra | movil | ninguno")
    region: str = Field(..., description="centro | norte | oeste | sur")

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 7,
                "monthly_charge": 58.23,
                "total_charges": 326.5,
                "support_tickets": 2,
                "late_payments": 1,
                "avg_monthly_usage_gb": 81.83,
                "num_products": 3,
                "customer_age": 53,
                "has_streaming": 0,
                "has_security_pack": 1,
                "is_promo": 1,
                "contract_type": "mensual",
                "payment_method": "transferencia",
                "internet_service": "cable",
                "region": "centro",
            }
        }
    }


class PrediccionOutput(BaseModel):
    churn_prediction: int
    churn_probability: float
    riesgo: str
    status: str


# ── Endpoints ────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "API de Predicción de Churn para AndesLink funcionando correctamente.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo_cargado": "model" in artifacts,
        "preprocesador_cargado": "preprocessor" in artifacts,
    }


@app.post("/predict", response_model=PrediccionOutput)
def predict(cliente: ClienteInput):
    if "model" not in artifacts or "preprocessor" not in artifacts:
        raise HTTPException(status_code=503, detail="Artefactos no cargados.")

    data = pd.DataFrame([{
        "tenure_months": cliente.tenure_months,
        "monthly_charge": cliente.monthly_charge,
        "total_charges": cliente.total_charges,
        "support_tickets": cliente.support_tickets,
        "late_payments": cliente.late_payments,
        "avg_monthly_usage_gb": cliente.avg_monthly_usage_gb,
        "num_products": cliente.num_products,
        "customer_age": cliente.customer_age,
        "has_streaming": cliente.has_streaming,
        "has_security_pack": cliente.has_security_pack,
        "is_promo": cliente.is_promo,
        "contract_type": cliente.contract_type.lower().strip(),
        "payment_method": cliente.payment_method.lower().strip(),
        "internet_service": cliente.internet_service.lower().strip(),
        "region": cliente.region.lower().strip(),
    }])

    try:
        X = artifacts["preprocessor"].transform(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en preprocesamiento: {str(e)}")

    try:
        prediction = int(artifacts["model"].predict(X)[0])
        probability = float(artifacts["model"].predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")

    if probability < 0.35:
        riesgo = "Bajo"
    elif probability < 0.65:
        riesgo = "Medio"
    else:
        riesgo = "Alto"

    return PrediccionOutput(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        riesgo=riesgo,
        status="success",
    )