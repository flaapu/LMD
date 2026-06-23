"""
gui.py – Interfaz gráfica para la API de predicción de churn (AndesLink)

Ejecutar con:
    streamlit run scr/gui.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8080/predict"

st.set_page_config(
    page_title="Predictor de Churn — AndesLink",
    page_icon="📡",
    layout="centered",
)

st.title("📡 Predictor de Abandono de Clientes")
st.caption("AndesLink Servicios Digitales S.A. — Sistema de retención")
st.divider()

st.subheader("Datos del cliente")

col1, col2 = st.columns(2)

with col1:
    tenure_months = st.number_input("Antigüedad (meses)", min_value=0, max_value=600, value=12)
    monthly_charge = st.number_input("Cargo mensual (USD)", min_value=0.0, max_value=500.0, value=60.0, step=0.5)
    total_charges = st.number_input("Cargos totales (USD)", min_value=0.0, max_value=50000.0, value=720.0, step=10.0)
    support_tickets = st.number_input("Tickets de soporte", min_value=0, max_value=50, value=1)
    late_payments = st.number_input("Pagos atrasados", min_value=0, max_value=50, value=0)
    avg_monthly_usage_gb = st.number_input("Consumo promedio mensual (GB)", min_value=0.0, max_value=1000.0, value=50.0)
    num_products = st.number_input("Cantidad de productos", min_value=1, max_value=20, value=2)
    customer_age = st.number_input("Edad del cliente", min_value=18, max_value=100, value=35)

with col2:
    contract_type = st.selectbox("Tipo de contrato", ["mensual", "anual", "bianual"])
    payment_method = st.selectbox("Método de pago", ["transferencia", "debito", "efectivo", "credito"])
    internet_service = st.selectbox("Servicio de internet", ["cable", "fibra", "movil", "ninguno"])
    region = st.selectbox("Región", ["centro", "norte", "oeste", "sur"])
    has_streaming = st.radio("¿Tiene streaming?", [0, 1], format_func=lambda x: "Sí" if x else "No", horizontal=True)
    has_security_pack = st.radio("¿Tiene pack de seguridad?", [0, 1], format_func=lambda x: "Sí" if x else "No", horizontal=True)
    is_promo = st.radio("¿Ingresó por promoción?", [0, 1], format_func=lambda x: "Sí" if x else "No", horizontal=True)

st.divider()

if st.button("🔍 Predecir riesgo de abandono", use_container_width=True, type="primary"):
    payload = {
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "num_products": num_products,
        "customer_age": customer_age,
        "has_streaming": has_streaming,
        "has_security_pack": has_security_pack,
        "is_promo": is_promo,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "region": region,
    }

    try:
        with st.spinner("Consultando API..."):
            response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            prob = result["churn_probability"]
            riesgo = result["riesgo"]
            prediction = result["churn_prediction"]

            st.subheader("Resultado")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Predicción", "🚨 Abandona" if prediction == 1 else "✅ Se queda")
            col_b.metric("Probabilidad", f"{prob:.1%}")
            col_c.metric("Riesgo", riesgo)

            if riesgo == "Alto":
                st.error("⚠️ Riesgo Alto — Activar campaña de retención inmediata.")
            elif riesgo == "Medio":
                st.warning("🔶 Riesgo Medio — Considerar contacto preventivo.")
            else:
                st.success("🟢 Riesgo Bajo — Cliente estable.")

            st.progress(prob, text=f"Probabilidad de abandono: {prob:.1%}")

        else:
            st.error(f"Error de la API ({response.status_code}): {response.json().get('detail', response.text)}")

    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API en http://localhost:8080. Verificá que el servidor esté corriendo.")
    except requests.exceptions.Timeout:
        st.error("La API tardó demasiado en responder.")
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")

st.divider()
st.caption("Modelo: Random Forest / Logistic Regression · MLflow · AndesLink MLOps")