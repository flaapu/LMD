"""
train.py – Entrenamiento y evaluación del modelo de churn

Entrena dos modelos (Regresión Logística + Random Forest), compara resultados con MLflow, serializa el mejor modelo.
"""

import os
import joblib
import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


# Configuración
PARAMS_PATH = "params.yaml"
PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"
REPORTS_PATH = "reports/figures"
MLFLOW_EXPERIMENT = "andeslink-churn"


def load_params(path: str = PARAMS_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data():
    """Carga los splits procesados."""
    train = pd.read_csv(f"{PROCESSED_PATH}/train.csv")
    test = pd.read_csv(f"{PROCESSED_PATH}/test.csv")

    X_train = train.drop(columns=["churn"]).values
    y_train = train["churn"].values
    X_test = test.drop(columns=["churn"]).values
    y_test = test["churn"].values

    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Calcula métricas clave para clasificación binaria de churn."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        # detectar churners reales
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Guarda la matriz de confusión como imagen."""
    os.makedirs(REPORTS_PATH, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No churn", "Churn"],
                yticklabels=["No churn", "Churn"])
    plt.title(f"Matriz de confusión – {model_name}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    path = f"{REPORTS_PATH}/confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def train_and_log(model, model_name: str, params: dict,
                  X_train, y_train, X_test, y_test) -> dict:
    """Entrena un modelo, evalúa y registra con MLflow."""

    with mlflow.start_run(run_name=model_name):

        # Entrenamiento
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Métricas
        metrics = compute_metrics(y_test, y_pred, y_prob)

        # Log en MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Matriz de confusión
        cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(cm_path)

        print(f"\n{'='*50}")
        print(f"Modelo: {model_name}")
        print(f"{'='*50}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")
        print(
            f"\n{classification_report(y_test, y_pred, target_names=['No churn', 'Churn'])}")

    return {"model": model, "metrics": metrics, "name": model_name}


def run():
    params = load_params()
    X_train, X_test, y_train, y_test = load_processed_data()
    rs = params["train"]["random_state"]

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []

    # ── Modelo 1: Regresión Logística ──────────────────────────
    lr_params = {"model": "LogisticRegression", "C": 1.0, "random_state": rs,
                 "class_weight": "balanced"}
    lr = LogisticRegression(C=lr_params["C"], random_state=rs,
                            class_weight="balanced", max_iter=500)
    results.append(train_and_log(lr, "Logistic Regression", lr_params,
                                 X_train, y_train, X_test, y_test))

    # ── Modelo 2: Random Forest ────────────────────────────────
    rf_params = {
        "model": "RandomForest",
        "n_estimators": params["model"]["n_estimators"],
        "max_depth": params["model"]["max_depth"],
        "random_state": rs,
        "class_weight": "balanced",
    }
    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        random_state=rs,
        class_weight="balanced",
    )
    results.append(train_and_log(rf, "Random Forest", rf_params,
                                 X_train, y_train, X_test, y_test))

    # ── Selección del mejor modelo por ROC-AUC ─────────────────
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    print(f"\n✅ Mejor modelo: {best['name']} "
          f"(ROC-AUC = {best['metrics']['roc_auc']:.4f})")

    # ── Serialización ──────────────────────────────────────────
    os.makedirs(MODELS_PATH, exist_ok=True)
    model_path = f"{MODELS_PATH}/best_model.joblib"
    joblib.dump(best["model"], model_path)
    print(f"   Modelo guardado en '{model_path}'")

    return best


if __name__ == "__main__":
    run()
