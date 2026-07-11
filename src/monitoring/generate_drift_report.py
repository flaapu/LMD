"""
generate_drift_report.py – Monitoreo de datos/modelo (Examen final)

Genera un reporte de drift con Evidently comparando:
  - Referencia: el dataset con el que se entrenó el modelo (data/raw/churn_sintetico.csv).
  - Actual/producción: un lote simulado de clientes "nuevos", construido a partir de la
    referencia pero con corrimientos deliberados en algunas variables, representando
    cómo podría verse la cartera de clientes 3 meses después de puesta en producción
    (suba de precios, cambio de mix de servicio de internet, más atrasos de pago, etc).

En un escenario real, el lote "actual" se reemplazaría por los requests reales
recibidos por la API (por ejemplo, logueados a un archivo o a una base de datos desde
el propio endpoint /predict).

Salidas:
  - data/monitoring/current_clients_sample.csv   → snapshot del lote "actual" simulado
  - reports/monitoring/drift_report.html          → reporte interactivo de Evidently
  - reports/monitoring/drift_summary.json         → resumen numérico para lectura rápida

Ejecutar con:
    python -m src.monitoring.generate_drift_report
"""

import json
import os

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

RAW_DATA_PATH = "data/raw/churn_sintetico.csv"
CURRENT_SAMPLE_PATH = "data/monitoring/current_clients_sample.csv"
REPORT_HTML_PATH = "reports/monitoring/drift_report.html"
SUMMARY_JSON_PATH = "reports/monitoring/drift_summary.json"

# Columnas de features que le llegan a la API (se excluye el target "churn",
# que en producción no se conoce al momento de la predicción).
FEATURE_COLS = [
    "tenure_months", "monthly_charge", "total_charges", "support_tickets",
    "late_payments", "avg_monthly_usage_gb", "contract_type", "payment_method",
    "internet_service", "has_streaming", "has_security_pack", "num_products",
    "region", "customer_age", "is_promo",
]


def load_reference() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    return df[FEATURE_COLS].copy()


def simulate_current_batch(reference: pd.DataFrame, n: int = 800, seed: int = 7) -> pd.DataFrame:
    """
    Simula un lote de clientes "nuevos" con drift respecto de la referencia.

    Los corrimientos elegidos representan cambios de negocio plausibles:
      - Suba general de tarifas (monthly_charge) → afecta directamente el riesgo de churn.
      - Migración de clientes hacia internet por fibra, en detrimento de cable.
      - Aumento de pagos atrasados (contexto macroeconómico más exigente).
      - Leve caída de la antigüedad promedio (más clientes nuevos entrando).
    """
    rng = np.random.default_rng(seed)
    sample = reference.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)

    # Suba de tarifas: +18% en promedio, con algo de ruido.
    sample["monthly_charge"] = (sample["monthly_charge"] * 1.18 +
                                 rng.normal(0, 2.0, size=n)).round(2).clip(lower=0)

    # Más atrasos de pago.
    sample["late_payments"] = (sample["late_payments"] +
                                rng.poisson(0.8, size=n)).astype(int)

    # Antigüedad promedio más baja (más clientes nuevos).
    sample["tenure_months"] = (sample["tenure_months"] *
                                rng.uniform(0.55, 0.85, size=n)).round().astype(int).clip(lower=0)

    # Migración hacia fibra: reasigno una porción de "cable" a "fibra".
    mask_cable = sample["internet_service"] == "cable"
    switch = rng.random(n) < 0.45
    sample.loc[mask_cable & switch, "internet_service"] = "fibra"

    # Total de cargos, coherente con la nueva tarifa y antigüedad.
    sample["total_charges"] = (sample["monthly_charge"] *
                                sample["tenure_months"].clip(lower=1)).round(2)

    return sample


def interpret(drift_result: dict) -> list[str]:
    """Arma observaciones legibles a partir del resultado de Evidently."""
    notes = []
    dataset_drift = drift_result["metrics"][0]["result"]["dataset_drift"]
    n_drifted = drift_result["metrics"][0]["result"]["number_of_drifted_columns"]
    n_total = drift_result["metrics"][0]["result"]["number_of_columns"]
    share = drift_result["metrics"][0]["result"]["share_of_drifted_columns"]

    notes.append(
        f"Drift a nivel dataset: {'SI' if dataset_drift else 'NO'} "
        f"({n_drifted}/{n_total} columnas con drift, {share:.0%})."
    )

    per_column = drift_result["metrics"][1]["result"]["drift_by_columns"]
    drifted_cols = [c for c, info in per_column.items() if info.get("drift_detected")]
    if drifted_cols:
        notes.append("Columnas con drift detectado: " + ", ".join(sorted(drifted_cols)))
    else:
        notes.append("No se detectó drift a nivel de columna individual.")

    return notes


def run():
    os.makedirs(os.path.dirname(CURRENT_SAMPLE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_HTML_PATH), exist_ok=True)

    reference = load_reference()
    current = simulate_current_batch(reference)
    current.to_csv(CURRENT_SAMPLE_PATH, index=False)
    print(f"Lote 'actual' simulado guardado en '{CURRENT_SAMPLE_PATH}' ({len(current)} filas).")

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(REPORT_HTML_PATH)
    print(f"Reporte HTML de Evidently guardado en '{REPORT_HTML_PATH}'.")

    result = report.as_dict()
    summary = {
        "dataset_drift": result["metrics"][0]["result"]["dataset_drift"],
        "number_of_drifted_columns": result["metrics"][0]["result"]["number_of_drifted_columns"],
        "number_of_columns": result["metrics"][0]["result"]["number_of_columns"],
        "share_of_drifted_columns": result["metrics"][0]["result"]["share_of_drifted_columns"],
        "interpretation": interpret(result),
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Resumen guardado en '{SUMMARY_JSON_PATH}'.\n")

    print("── Interpretación ──")
    for note in summary["interpretation"]:
        print(f"  • {note}")


if __name__ == "__main__":
    run()
