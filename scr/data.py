"""
data.py – Preprocesamiento del dataset de churn

Responsabilidades:
  - Carga del dataset raw
  - Separación features / target
  - Encoding de variables categóricas
  - Escalado de variables numéricas continuas
  - Split train / test
  - Persistencia del dataset procesado y del pipeline de preprocesamiento
"""

import os
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
PARAMS_PATH = "../params.yaml"
RAW_DATA_PATH = "../data/raw/churn_sintetico.csv"
PROCESSED_PATH = "../data/processed"
MODELS_PATH = "../models"

# lee los archivos yaml de configuracion


def load_params(path: str = PARAMS_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Definición de columnas
TARGET = "churn"

NUMERIC_COLS = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "support_tickets",
    "late_payments",
    "avg_monthly_usage_gb",
    "num_products",
    "customer_age",
]

BINARY_COLS = [
    "has_streaming",
    "has_security_pack",
    "is_promo",
]

CATEGORICAL_COLS = [
    "contract_type",
    "payment_method",
    "internet_service",
    "region",
]

# Procesamiento
#


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el dataset raw."""
    df = pd.read_csv(path)
    print(
        f"[data_prep] Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def split_features_target(df: pd.DataFrame):
    """Separa features (X) y target (y)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el pipeline de preprocesamiento:
      - Numéricas: StandardScaler
      - Binarias: passthrough (sin transformación)
      - Categóricas: OneHotEncoder (drop='first' para evitar multicolinealidad)
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(drop="first",
         handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_COLS),
        ("bin", "passthrough", BINARY_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ])

    return preprocessor


def run(params: dict = None):
    """Pipeline completo de preprocesamiento."""

    if params is None:
        params = load_params()

    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    # 1. Carga
    df = load_data()

    # 2. Split features / target
    x, y = split_features_target(df)

    # 3. Train / test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # mantiene proporción de churn en ambos splits
    )
    print(f"[data_prep] Train: {x_train.shape[0]} | Test: {x_test.shape[0]}")
    print(
        f"[data_prep] Churn rate train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    # 4. Ajustar preprocesador SOLO sobre train
    preprocessor = build_preprocessor()
    x_train_proc = preprocessor.fit_transform(x_train)
    x_test_proc = preprocessor.transform(x_test)

    # 5. Recuperar nombres de columnas del output
    ohe_features = preprocessor.named_transformers_["cat"] \
        .named_steps["onehot"].get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERIC_COLS + BINARY_COLS + ohe_features

    # 6. Guardar datasets procesados
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    pd.DataFrame(x_train_proc, columns=feature_names).assign(churn=y_train.values) \
        .to_csv(f"{PROCESSED_PATH}/train.csv", index=False)
    pd.DataFrame(x_test_proc, columns=feature_names).assign(churn=y_test.values) \
        .to_csv(f"{PROCESSED_PATH}/test.csv", index=False)

    print(f"[data_prep] Datos guardados en '{PROCESSED_PATH}/'")

    # 7. Guardar el preprocesador para inference
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(preprocessor, f"{MODELS_PATH}/preprocessor.joblib")
    print(
        f"[data_prep] Preprocesador guardado en '{MODELS_PATH}/preprocessor.joblib'")

    return x_train_proc, x_test_proc, y_train.values, y_test.values, feature_names


if __name__ == "__main__":
    run()
