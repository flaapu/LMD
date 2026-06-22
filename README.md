# LMD
Laboratorio de Minería de Datos

Contexto de negocio
AndesLink Servicios Digitales S.A. comercializa planes de suscripción para servicios digitales orientados a
consumidores finales. Durante los últimos trimestres, la empresa ha detectado una tasa creciente de
cancelación voluntaria de clientes. El directorio entiende que la pérdida de clientes no solo afecta los ingresos
recurrentes, sino también el costo de adquisición de nuevos usuarios, la estabilidad del flujo de caja y la
eficiencia de sus campañas comerciales.
Con el objetivo de tomar decisiones más oportunas, la empresa desea contar con un modelo de Machine
Learning capaz de estimar la probabilidad de churn a partir de variables de comportamiento, antigüedad,
facturación y relación con el servicio. Además, exige que la solución no quede limitada al entrenamiento del
modelo: debe poder ejecutarse localmente, exponerse mediante una API, consumirse desde una interfaz
gráfica simple y contar con monitoreo técnico y de datos.
La empresa contrata al alumno como responsable de construir una solución end to end en un entorno local,
con trazabilidad, reproducibilidad y organización de proyecto semejantes a las de un escenario profesional.


Guía de como ejecutar el código

1. Como primera y única instancia:
ejecutá: pip install -r requirements.txt

2. Ejecutar data.py 
Como segunda(o primera) instancia, correr el data.py
Que hace este archivo? lee el csv, limpia, escala, codifica y crea los archivos train.csv y test.csv en la carpeta processed. Tambien crea y guarda el archivo preprocessor.joblib

En caso de que no encuentres los archivos creados en la carpeta, indica que algo salió mal. 

3. Ejecutar train.py 
Lee los csv creadosen /processed, entrena con regresion logistica y rf, compara cual es mejor y guarda el dato en models/best_model.joblib

por consola debería de mostrar el recal y el ROC-AUC.

4. Ejecutar finalmente el archivo notebook/eda.ipynb
Ayuda como documentación de decisiones.
La idea de su ejecucuion es que se generen los gráficos para posterior análisis.

