# API de Predicción de Churn (Fugas de Clientes) 🚀

Este proyecto implementa un pipeline completo de Machine Learning (MLOps) enfocado en predecir el Churn de clientes. Cuenta con etapas automatizadas de procesamiento de datos, entrenamiento del modelo (trackeado con MLflow), pruebas unitarias automáticas con Pytest y empaquetamiento en un contenedor Docker usando FastAPI.

## 📦 Estructura del Proyecto
* `scr/data.py`: Carga y preprocesamiento del dataset.
* `scr/train.py`: Entrenamiento del modelo de Regresión Logística y registro en MLflow.
* `scr/app.py`: API REST construida con FastAPI para servir las predicciones.
* `tests/test_api.py`: Pruebas de integración automatizadas para validar la API.
* `Dockerfile`: Configuración del contenedor Docker para producción.

---

## 🛠️ Instrucciones para Ejecutar el Proyecto con Docker

Para compilar la imagen y levantar el servicio sin necesidad de configurar Python localmente, sigue estos pasos:

### 1. Construir la Imagen Docker
Este comando instalará las dependencias necesarias, ejecutará el pipeline de datos, entrenará el modelo de ML y correrá los tests unitarios automáticos antes de finalizar la construcción:

docker build -t churn-service .

### 2. Levantar el Contenedor (Servidor API)
Una vez construida la imagen con éxito, enciende el contenedor mapeando el puerto 8080:

docker run -p 8080:8080 churn-service

### Cómo Probar la API en Vivo
Cuando el contenedor esté corriendo, abre tu navegador web e ingresa a la documentación interactiva:
👉 http://localhost:8080/docs

Ejemplo de Prueba (Endpoint /predict)
El modelo espera un vector de entrada con las 22 características (features) preprocesadas del cliente. Puedes usar el siguiente JSON de ejemplo dentro del botón "Try it out" -> "Execute":

[
  7.0,     // tenure_months (Meses de antigüedad)
  58.23,   // monthly_charge (Cargo mensual)
  326.5,   // total_charges (Cargos totales)
  2.0,     // support_tickets (Tickets de soporte abiertos)
  1.0,     // late_payments (Pagos atrasados)
  81.83,   // avg_monthly_usage_gb (Consumo de GB promedio)
  1.0, 0.0, 0.0, // contract_type (One-Hot: Mensual, Anual, Bianual)
  0.0, 1.0, 0.0, 0.0, // payment_method (One-Hot: Transferencia, Débito, Efectivo, Crédito)
  1.0, 0.0, 0.0, 0.0, // internet_service (Cable, Fibra, Móvil, Ninguno)
  0.0,     // has_streaming (0 = No, 1 = Sí)
  1.0,     // has_security_pack
  3.0,     // num_products (Cantidad de productos contratados)
  1.0, 0.0, 0.0, 0.0, // region (Centro, Norte, Oeste, Sur)
  53.0,    // customer_age (Edad del cliente)
  1.0      // is_promo (Si entró por promoción)
]
NOTA: La FastAPI no acepta comentarios dentro del cuadro de texto, asi que eliminar los comentarios para testear. 

[7.0, 58.23, 326.5, 2.0, 1.0, 81.83, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 53.0]

La API responderá en tiempo real con la predicción (0 si el cliente se queda, 1 si se da de baja) y su respectiva probabilidad.