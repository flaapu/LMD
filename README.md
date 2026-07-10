# API de Predicción de Churn (Fugas de Clientes) 🚀

Este proyecto implementa un pipeline completo de Machine Learning (MLOps) enfocado en predecir el Churn de clientes. Cuenta con etapas automatizadas de procesamiento de datos, entrenamiento del modelo (trackeado con MLflow), pruebas unitarias automáticas con Pytest y empaquetamiento en un contenedor Docker usando FastAPI.

## 📦 Estructura del Proyecto
* `src/data.py`: Carga, limpieza y preprocesamiento del dataset.
* `src/train.py`: Entrenamiento del modelo de Regresión Logística y registro en MLflow.
* `src/app.py`: API REST construida con FastAPI para servir las predicciones en producción.
* `test/test_api.py`: Pruebas automatizadas con Pytest para validar el comportamiento del backend.
* `Dockerfile` & `docker-compose.yml`: Configuración y orquestación del contenedor local.
* `environment.yml`: Archivo de configuración del entorno de Conda.

---

## 🛠️ Instrucciones para Ejecutar el Proyecto con Docker Compose

1. Entrenar el modelo (una sola vez, o cuando cambien los datos):
   python src/data.py
   python src/train.py

2. Correr los tests (opcional pero recomendado antes de desplegar):

  **Windows (PowerShell):**
  $env:PYTHONPATH="." ; pytest

  **Mac/Linux (Bash):**
  PYTHONPATH=. pytest

3. Levantar la API y la GUI con Docker Compose:
   docker compose up --build

###  Cómo Probar la API en Vivo 
Cuando el contenedor esté corriendo, abra su navegador web e ingrese a la documentación interactiva de Swagger UI:
http://localhost:8080/docs  

Ejemplo de Prueba (Endpoint /predict)
La API espera un objeto JSON con los datos crudos del cliente (15 campos: antigüedad, cargos, 
tipo de contrato, región, etc.), no una lista de features preprocesadas — el preprocesamiento 
lo aplica la propia API antes de invocar al modelo.
Instrucciones para Pruebas: Presione el botón "Try it out", borre el contenido del cuadro de texto 
y copie el siguiente JSON estructurado (sin comentarios) para testear la inferencia en tiempo real:
JSON
{
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
  "region": "centro"
}

Al presionar el botón azul "Execute", la API responderá con un formato consistente que incluye la predicción final (0 si el cliente mantiene el servicio, 1 si se predice abandono) junto a su respectiva probabilidad estimada.