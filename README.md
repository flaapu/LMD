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

Para compilar las imágenes, ejecutar el pipeline completo e iniciar toda la infraestructura de la solución en un entorno local reproducible, ejecute el siguiente comando en la raíz del proyecto:

docker compose up --build

Este comando instalará las dependencias requeridas, ejecutará el pipeline de datos, entrenará el modelo, validará la API corriendo los tests automatizados y finalmente levantará el servidor web escuchando de forma exclusiva en el puerto 8080.

docker build -t churn-service .

###  Cómo Probar la API en VivoCuando el contenedor esté corriendo, abra su navegador web e ingrese a la documentación interactiva de Swagger UI:
http://localhost:8080/docs  

Ejemplo de Prueba (Endpoint /predict)   
La API define un contrato estricto de validación que requiere un objeto JSON con la clave "features" que contenga una lista de exactamente 22 características preprocesadas del cliente.  
Instrucciones para Pruebas: Presione el botón "Try it out", borre el contenido del cuadro de texto y copie el siguiente JSON estructurado (sin comentarios) para testear la inferencia en tiempo real:  

JSON
{
  "features": [7.0, 58.23, 326.5, 2.0, 1.0, 81.83, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 53.0]
}

Al presionar el botón azul "Execute", la API responderá con un formato consistente que incluye la predicción final (0 si el cliente mantiene el servicio, 1 si se predice abandono) junto a su respectiva probabilidad estimada.