import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    recall_score,
)
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
import numpy as np
import mlflow

def baseline_persistencia_categorica(
    df: pd.DataFrame,
    target_col: str,
    horizon: int
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    Implementa un modelo baseline de persistencia para variables categóricas.
    Predice que la tendencia será igual a la de `horizon` días atrás, permitiendo solo horizontes de 7 o 30 días.

    :param df: DataFrame con los datos de mercado.
    :type df: pd.DataFrame
    :param target_col: Nombre de la columna objetivo (valores posibles: -1 para bajista, 0 para neutra, 1 para alcista).
    :type target_col: str
    :param horizon: Horizonte de predicción (solo acepta valores 7 o 30 días).
    :type horizon: int
    :raises ValueError: Si el horizonte de predicción no es 7 o 30 días.
    :return: DataFrame con las predicciones añadidas, matriz de confusión, accuracy, y recall.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, float, float, float]
    """
    if horizon not in [7, 30]:
        raise ValueError("El horizonte de predicción solo puede ser 7 o 30 días.")

    df = df.copy()
    df["baseline_pred"] = df[target_col].shift(horizon)
    df = df.dropna()  # Elimina los valores nulos creados por el shift

    cm = pd.DataFrame(
        confusion_matrix(df[target_col], df["baseline_pred"], labels=[-1, 1]),
        index=["Real -1", "Real 1"],
        columns=["Pred -1", "Pred 1"],
    )

    # Cálculo de las métricas de clasificación
    accuracy = accuracy_score(df[target_col], df["baseline_pred"])
    recall = recall_score(
        df[target_col], df["baseline_pred"], average="weighted"
    )  # Average weighted para manejo de clases desbalanceadas

    return df, cm, accuracy, recall


def baseline_persistencia_numerica(
    df: pd.DataFrame, target_col: str, horizon: int
) -> Tuple[pd.DataFrame, float, float, float, float]:
    """
    Implementa un modelo baseline de persistencia para variables numéricas.
    Predice que el valor de la variable será igual al de `horizon` días atrás.

    :param df: DataFrame con los datos de mercado.
    :type df: pd.DataFrame
    :param target_col: Nombre de la columna objetivo numérica.
    :type target_col: str
    :param horizon: Horizonte de predicción (solo acepta valores 7 o 30 días).
    :type horizon: int
    :raises ValueError: Si el horizonte de predicción no es 7 o 30 días.
    :return: DataFrame con las predicciones añadidas, MAE, MSE, RMSE, y R^2.
    :rtype: Tuple[pd.DataFrame, float, float, float, float]
    """
    if horizon not in [7, 30]:
        raise ValueError("El horizonte de predicción solo puede ser 7 o 30 días.")

    df = df.copy()
    df["baseline_pred"] = df[target_col].shift(horizon)
    df = df.dropna()  # Elimina valores nulos creados por el shift
    mae = mean_absolute_error(df[target_col], df["baseline_pred"])
    mse = mean_squared_error(df[target_col], df["baseline_pred"])
    rmse = np.sqrt(mse)  # RMSE
    r2 = r2_score(df[target_col], df["baseline_pred"])  # R^2

    return df, mae, mse, rmse, r2


if __name__ == "__main__":
    mlflow.set_tracking_uri(r"file:mlruns")
    mlflow.set_experiment("Baseline")

    # Datos para 7 días
    df = pd.read_parquet(r"../data/parquet/data_7d.parquet")

    print("BASELINE PARA 7 DIAS:\n")

    # Categórica
    with mlflow.start_run(run_name=f"Baseline 7d categorica"):
        _, cm, accuracy, recall = baseline_persistencia_categorica(
            df, "tendencia_futura_7d", 7
        )
        print("Matriz de Confusión:\n", cm)
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")

        # Log confusion matrix as a figure or artifact
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')

        mlflow.log_metrics({"accuracy": accuracy, "recall": recall})
        mlflow.log_figure(fig, "confusion_matrix.png")

    # Numérica
    with mlflow.start_run(run_name=f"Baseline 7d regresion"):
        _, mae, mse, rmse, r2 = baseline_persistencia_numerica(df, "pct_cambio_7d", 7)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")
        print('\n\n')

        mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": rmse, "r2":r2})

    # Datos para 30 días
    df = pd.read_parquet(r"../data/parquet/data_30d.parquet")

    print("BASELINE PARA 30 DIAS:\n")

    # Categórica
    with mlflow.start_run(run_name=f"Baseline 30d categorica"):
        _, cm, accuracy, recall = baseline_persistencia_categorica(
            df, "tendencia_futura_30d", 30
        )
        print("Matriz de Confusión:\n", cm)
        print(f"Accuracy: {accuracy}")
        print(f"Recall: {recall}")

        # Log confusion matrix as a figure or artifact
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')

        mlflow.log_metrics({"accuracy": accuracy, "recall": recall})
        mlflow.log_figure(fig, "confusion_matrix.png")

    # Numérica
    with mlflow.start_run(run_name=f"Baseline 30d regresion"):
        _, mae, mse, rmse, r2 = baseline_persistencia_numerica(df, "pct_cambio_30d", 30)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}")

        mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": rmse, "r2": r2})