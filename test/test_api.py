"""
test_api.py – Pruebas automáticas de la API de churn (AndesLink)

Ejecutar con:
    PYTHONPATH=. pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from scr.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


CLIENTE_VALIDO = {
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


# ── Health ───────────────────────────────────────────────────

def test_home_responde_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelo_cargado"] is True
    assert data["preprocesador_cargado"] is True


# ── Inferencia correcta ───────────────────────────────────────

def test_predict_cliente_valido(client):
    response = client.post("/predict", json=CLIENTE_VALIDO)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["riesgo"] in ["Bajo", "Medio", "Alto"]
    assert data["status"] == "success"


def test_predict_devuelve_probabilidad_float(client):
    response = client.post("/predict", json=CLIENTE_VALIDO)
    assert response.status_code == 200
    assert isinstance(response.json()["churn_probability"], float)


def test_predict_cliente_estable(client):
    cliente = CLIENTE_VALIDO.copy()
    cliente.update({"tenure_months": 60, "contract_type": "bianual",
                    "support_tickets": 0, "late_payments": 0})
    response = client.post("/predict", json=cliente)
    assert response.status_code == 200


# ── Validación de entrada ─────────────────────────────────────

def test_payload_vacio_retorna_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_edad_invalida_retorna_422(client):
    cliente = CLIENTE_VALIDO.copy()
    cliente["customer_age"] = 5
    response = client.post("/predict", json=cliente)
    assert response.status_code == 422


def test_cargo_negativo_retorna_422(client):
    cliente = CLIENTE_VALIDO.copy()
    cliente["monthly_charge"] = -10.0
    response = client.post("/predict", json=cliente)
    assert response.status_code == 422


def test_campo_faltante_retorna_422(client):
    cliente = CLIENTE_VALIDO.copy()
    del cliente["contract_type"]
    response = client.post("/predict", json=cliente)
    assert response.status_code == 422