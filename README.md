# API de Predicción de Churn (Fugas de Clientes) 🚀

Proyecto de MLOps local para **AndesLink Servicios Digitales S.A.**: predice el abandono (churn) de clientes a partir de sus datos de servicio. Cubre el ciclo completo: entrenamiento y tracking con MLflow/DVC, una API de inferencia con FastAPI, una GUI en Streamlit, contenedorización con Docker Compose, y monitoreo técnico (Prometheus + Grafana) y de datos (Evidently).

## 📦 Estructura del Proyecto
* `src/data.py`: Carga, limpieza y preprocesamiento del dataset. Genera y persiste el preprocesador.
* `src/train.py`: Entrenamiento (Regresión Logística + Random Forest) y registro en MLflow.
* `src/app.py`: API REST con FastAPI. Sirve predicciones y expone métricas técnicas en `/metrics`.
* `src/gui.py`: Interfaz gráfica en Streamlit que consume la API.
* `src/config.py`: Configuración centralizada (rutas de modelo, puerto, URL de la API) vía variables de entorno.
* `src/monitoring/generate_drift_report.py`: Genera el reporte de drift de datos con Evidently.
* `test/test_api.py`: Pruebas automatizadas con Pytest.
* `monitoring/prometheus/`: Configuración de scraping de Prometheus.
* `monitoring/grafana/`: Provisioning de datasource y dashboard de Grafana.
* `Dockerfile`, `Dockerfile.gui` & `docker-compose.yml`: Contenedorización y orquestación local.
* `environment.yml`: Entorno de Conda para desarrollo local (fuera de Docker).

---

## 🗺️ Arquitectura del despliegue local

```
 Usuario
   │
   ▼
┌───────────────────────┐        POST /predict        ┌───────────────────────────┐
│   GUI (Streamlit)      │ ───────────────────────────▶ │   API (FastAPI)            │
│   contenedor: gui       │                              │   contenedor: api           │
│   puerto host: 8501     │ ◀─────────────────────────── │   puerto host: 8080         │
└───────────────────────┘        JSON de respuesta       └──────────┬─────────────────┘
                                                                     │  predicción            │  GET /metrics
                                                                     ▼                        ▼
                                                     ┌────────────────────────┐   ┌──────────────────────┐
                                                     │ models/*.joblib         │   │  Prometheus            │
                                                     │ (modelo + preprocesador)│   │  puerto host: 9090      │
                                                     └────────────────────────┘   └──────────┬───────────┘
                                                                                              │ scrape 5s
                                                                                              ▼
                                                                                   ┌──────────────────────┐
                                                                                   │  Grafana               │
                                                                                   │  puerto host: 3000      │
                                                                                   └──────────────────────┘
```

Dentro de Docker Compose, la GUI le habla a la API por el nombre del servicio (`http://api:8080/predict`), no por `localhost` — esa URL se inyecta vía la variable de entorno `API_URL`, que `src/gui.py` lee con `os.getenv`. Prometheus scrapea el endpoint `/metrics` de la API cada 5 segundos, y Grafana lee esas métricas desde Prometheus para armar los tableros.

El monitoreo de datos (Evidently) es un proceso aparte, no un servicio siempre-activo: se corre bajo demanda con `src/monitoring/generate_drift_report.py`, comparando el dataset de entrenamiento contra un lote de datos "actuales" (ver sección de Monitoreo más abajo).

---

## 🛠️ Instrucciones para Ejecutar el Proyecto

### 1. Entrenar el modelo (una sola vez, o cuando cambien los datos)
```bash
python src/data.py
python src/train.py
```

### 2. Correr los tests
```bash
# Windows (PowerShell)
$env:PYTHONPATH="." ; pytest -v

# Mac/Linux (Bash)
PYTHONPATH=. pytest -v
```

### 3. Levantar todo el stack con Docker Compose
```bash
docker compose up --build
```

Esto levanta 4 servicios:

| Servicio    | URL                              | Descripción                              |
|-------------|-----------------------------------|-------------------------------------------|
| API         | http://localhost:8080/docs        | Swagger UI para probar `/predict`         |
| GUI         | http://localhost:8501             | Formulario Streamlit                      |
| Prometheus  | http://localhost:9090             | Métricas técnicas crudas y explorador     |
| Grafana     | http://localhost:3000             | Dashboard (usuario/clave: `admin`/`admin`, o entrar como anónimo) |

### Ejecutar solo la API (sin Compose)
```bash
docker build -t churn-api .
docker run -p 8080:8080 churn-api
```

### Ejecutar la GUI localmente (fuera de Docker)
```bash
pip install -r requirements.txt
streamlit run src/gui.py
```
Sin `API_URL` definida, usa `http://localhost:8080/predict` por defecto (la API debe estar corriendo en el puerto 8080).

---

## 📡 Cómo probar la API en vivo

Con la API corriendo, abrí Swagger UI: [http://localhost:8080/docs](http://localhost:8080/docs)

> ⚠️ Nota importante: `/predict` sólo acepta método **POST**. Si escribís `localhost:8080/predict` directo en la barra del navegador vas a ver `{"detail":"Method Not Allowed"}` — eso es correcto, el navegador hace GET. Usá Swagger (`/docs`), la GUI, o un cliente HTTP (curl, Postman) que permita mandar POST.

### Contrato del endpoint `POST /predict`
La API recibe los **datos crudos del cliente** (15 campos, los mismos que completa la GUI), no una lista de features preprocesadas — el preprocesamiento lo aplica la propia API internamente. Las variables categóricas están restringidas con `Enum`; cualquier valor fuera de las opciones válidas es rechazado con `422`.

```json
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
```

Valores válidos: `contract_type` (mensual, anual, bianual) · `payment_method` (transferencia, debito, efectivo, credito) · `internet_service` (cable, fibra, movil, ninguno) · `region` (centro, norte, oeste, sur).

Otros endpoints: `GET /` (estado general) · `GET /health` (chequeo de artefactos cargados) · `GET /metrics` (métricas técnicas en formato Prometheus).

---

## 📊 Monitoreo técnico (Prometheus + Grafana)

La API se instrumenta con `prometheus-fastapi-instrumentator`, que expone automáticamente `/metrics` con contadores y latencias por endpoint (sin necesidad de tocar la lógica de negocio).

1. Levantar el stack: `docker compose up --build`.
2. Generar tráfico de prueba (llamar `/predict` varias veces desde Swagger, la GUI, o un loop con curl).
3. Ver métricas crudas: [http://localhost:9090](http://localhost:9090) → pestaña *Graph*, por ejemplo con la query `rate(http_requests_total[1m])`.
4. Ver el dashboard armado: [http://localhost:3000](http://localhost:3000) → dashboard **"Churn API — Monitoreo Técnico"** (se carga solo, vía provisioning). Incluye:
   - Requests por segundo, por endpoint.
   - Latencia p95.
   - Tasa de error (respuestas 5xx).
   - Requests en curso (in-progress).
   - Requests totales por código de status.

**Señales que ameritarían acción correctiva:**
* Suba sostenida de la tasa de error 5xx → revisar logs de la API, posible problema con los artefactos del modelo o con el preprocesador.
* Aumento sostenido de la latencia p95 → revisar carga del contenedor, tamaño de payloads, o necesidad de escalar recursos.
* Caída a cero de requests cuando se espera tráfico → posible caída de la GUI o de la conectividad entre servicios (chequear `docker compose ps` y `/health`).

---

## 🔎 Monitoreo de datos y drift (Evidently)

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m src.monitoring.generate_drift_report
```

El script:
1. Toma el dataset de entrenamiento (`data/raw/churn_sintetico.csv`) como **referencia**.
2. Simula un lote de clientes "**actuales**" (`data/monitoring/current_clients_sample.csv`), aplicando corrimientos deliberados que representan un escenario de negocio plausible 3 meses después del despliegue: suba de tarifas (+18%), más atrasos de pago, menor antigüedad promedio (más altas recientes) y migración de clientes de cable hacia fibra.

   > En producción real, este lote "actual" se reemplazaría por los requests reales recibidos por `/predict` (logueados a un archivo o base de datos desde la propia API).
3. Genera un reporte interactivo en `reports/monitoring/drift_report.html` (abrilo en el navegador) y un resumen numérico en `reports/monitoring/drift_summary.json`.

**Hallazgo de la corrida de referencia incluida en este repo:** se detecta drift en 5 de 15 columnas (`monthly_charge`, `tenure_months`, `late_payments`, `total_charges`, `internet_service`) — coherente con los corrimientos simulados. El drift no llega a superar el umbral de "dataset drift" global (33% de columnas, contra 50% de umbral por defecto de Evidently), pero el drift en `monthly_charge` y `internet_service` es especialmente relevante porque son variables con peso conocido en el riesgo de abandono.

**Señales que ameritarían acción correctiva:**
* Drift sostenido en variables con alto peso en el modelo (tarifa, antigüedad, tipo de servicio) → evaluar reentrenamiento con datos más recientes.
* Drift a nivel dataset (>50% de columnas) → pausar confianza en las predicciones hasta validar el modelo contra el nuevo contexto.
* Caída marcada en calidad de datos (nulos, valores fuera de rango) → revisar la fuente de datos antes de reentrenar.

---

## 🧪 Correr las pruebas manualmente (fuera de Docker)
```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -v
```

---

## ⚠️ Notas y limitaciones conocidas
* El `Dockerfile` de la API no entrena ni testea durante el `build` (esos pasos están comentados intencionalmente): se asume que el modelo ya fue entrenado y versionado (`models/*.joblib` se incluyen en el repo), separando así entrenamiento/validación de serving, como recomienda buena práctica de MLOps.
* El lote "actual" de Evidently es simulado, no proviene de tráfico real de producción (no hay una base de datos de logging de requests en este proyecto). Documentado explícitamente para que quede claro en la defensa técnica.
* No hay pruebas automatizadas de integración end-to-end entre la GUI y la API (solo pruebas unitarias de la API).
