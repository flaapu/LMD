from scr.app import app
import sys
import os
from fastapi.testclient import TestClient

# 1. Configuramos las rutas primero para que Python sepa dónde buscar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Ahora sí, importamos la app de forma segura
client = TestClient(app)


def test_read_main():
    """Prueba que el endpoint raíz responda correctamente"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "API de Predicción de Churn funcionando correctamente."
    }


def test_predict_without_model():
    """Prueba el comportamiento si el modelo no existiera o falla con datos vacíos"""
    response = client.post("/predict", json=[])
    assert response.status_code in [400, 404]
