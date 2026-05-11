"""
data_prep.py – Preprocesamiento del dataset de churn
AndesLink Servicios Digitales S.A. | ISTEA · Laboratorio de Minería de Datos

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
PARAMS_PATH = "params.yaml"
RAW_DATA_PATH = "data/raw/churn_sintetico.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


def load_params(path: str = PARAMS_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Definición de columnas
# ──────────────────────────────────────────────
TARGET = "churn"

NUMERIC_COLS = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "support_tickets",
    "late_payments",
    "avg_monthly_usage_gb",
    "customer_age",
]

# Binarias: ya son 0/1, no necesitan escala ni encoding
BINARY_COLS = [
    "has_streaming",
    "has_security_pack",
    "is_promo",
    "num_products",  # ordinal tratado como numérica
]

CATEGORICAL_COLS = [
    "contract_type",
    "payment_method",
    "internet_service",
    "region",
]


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
    X, y = split_features_target(df)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # mantiene proporción de churn en ambos splits
    )
    print(f"[data_prep] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(
        f"[data_prep] Churn rate train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    # 4. Ajustar preprocesador SOLO sobre train
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 5. Recuperar nombres de columnas del output
    ohe_features = preprocessor.named_transformers_["cat"] \
        .named_steps["onehot"].get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERIC_COLS + BINARY_COLS + ohe_features

    # 6. Guardar datasets procesados
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    pd.DataFrame(X_train_proc, columns=feature_names).assign(churn=y_train.values) \
        .to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    pd.DataFrame(X_test_proc, columns=feature_names).assign(churn=y_test.values) \
        .to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    print(f"[data_prep] Datos guardados en '{PROCESSED_DIR}/'")

    # 7. Guardar el preprocesador para inference
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.joblib")
    print(
        f"[data_prep] Preprocesador guardado en '{MODELS_DIR}/preprocessor.joblib'")

    return X_train_proc, X_test_proc, y_train.values, y_test.values, feature_names


if __name__ == "__main__":
    run()
