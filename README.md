## Laboratorio MLOps - Predicción de Churn (LMD)
Este proyecto implementa un pipeline de Machine Learning estructurado y reproducible para predecir el Churn (abandono) de clientes, utilizando buenas prácticas de MLOps como control de versiones de código y tracking de experimentos.

# Estructura del Pipeline
El proyecto está dividido en etapas modulares para garantizar la mantenibilidad:

1. Preprocesamiento (scr/data.py): Limpieza de datos crudos, imputación, ingeniería de features y división del dataset en sets de entrenamiento y testeo.

2. Entrenamiento (scr/train.py): Entrenamiento competitivo de modelos, optimización de hiperparámetros, evaluación de métricas y registro automático en MLflow.

# Requisitos e Instalación
Para replicar este entorno localmente, asegúrese de contar con Python 3.12 instalado y ejecute el siguiente comando para instalar las dependencias requeridas:

pip install dvc mlflow pandas numpy scikit-learn joblib pyyaml

# Ejecución del Proyecto
Para correr el pipeline completo de punta a punta de forma nativa, ejecute los siguientes comandos en orden desde la raíz del proyecto:

1. Ejecutar el preprocesamiento de datos
python scr/data.py

2. Ejecutar el entrenamiento y registro de modelos
python scr/train.py

# Resultados del Experimento
Durante la fase de entrenamiento, se evaluaron dos arquitecturas de modelos sobre el set de datos procesado. Los resultados obtenidos fueron los siguientes:

- Métrica: Regresión Logística (Ganador) | Random Forest
- Accuracy: 0.6740 | 0.6910
- Precision: 0.5145 | 0.5405
- Recall: 0.7294 | 0.6088
- F1-Score: 0.6034 | 0.5726
- ROC-AUC: 0.7580 | 0.7366

Modelo Seleccionado
El mejor modelo elegido por el pipeline fue Logistic Regression con un ROC-AUC de 0.7580.

Justificación: Aunque Random Forest obtuvo un accuracy ligeramente mayor, la Regresión Logística demostró un Recall significativamente superior (0.7294 vs 0.6088). En un problema de Churn, detectar a tiempo la mayor cantidad de clientes en riesgo de abandono (minimizar falsos negativos) es crítico para el negocio, haciendo de este modelo la opción óptima.

El artefacto final se encuentra exportado y listo para producción en models/best_model.joblib.

# Tracking con MLflow
Todas las métricas, parámetros y matrices de confusión fueron registrados localmente en el servidor de experimentos. Para levantar la interfaz interactiva de MLflow y explorar los resultados visualmente, ejecute:

mlflow ui

Luego, ingrese a http://localhost:5000 desde su navegador web.