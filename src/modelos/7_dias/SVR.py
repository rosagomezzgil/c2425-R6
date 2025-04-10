# Este archivo implementa un modelo de regresión utilizando Support Vector Regression (SVR) para predecir el porcentaje de cambio
# de la tendencia del Bitcoin en un período de 7 días. El proceso abarca desde la carga de los datos, el preprocesamiento,
# la optimización de hiperparámetros hasta la evaluación y visualización de los resultados. El modelo es evaluado en base a
# métricas estándar como MSE, RMSE, MAE y R², y se utiliza MLflow para el seguimiento y registro de experimentos.
# Se debe tener el data_7d.parquet (parquet_merged_data.py en transformacion)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from itertools import product
import matplotlib.pyplot as plt
import mlflow
import random

# Fijar semillas
def fijar_semillas(seed: int = 123):
    """
    Fija las semillas para reproducibilidad

    :param seed: Semilla para las bibliotecas random, numpy y sklearn
    """
    np.random.seed(seed)
    random.seed(seed)

# 1. Cargar los datos
def cargar_datos(parquet_file: str):
    """
    Carga los datos de un fichero parquet a un DataFrame de pandas

    :param parquet_file: URL del fichero parquet

    :return: pd.DataFrame
    """
    df = pd.read_parquet(parquet_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Asegurar que la fecha esté en formato datetime
    df.set_index('timestamp', inplace=True)
    return df


# Función para preprocesar los datos
def preprocesar_datos(df: pd.DataFrame):
    """
        Preprocesa los datos eliminando columnas innecesarias y aplicando escalado.

        :param df: DataFrame con los datos a preprocesar.

        :returns: Tupla con:
            - X_scaled: Features escalados.
            - y: Variable objetivo (pct_cambio_7d).
            - scaler: Objeto StandardScaler entrenado con los datos.
    """
    df = df.drop(
        columns=["tendencia_futura_7d", "pct_cambio_30d", "tendencia_futura_30d"],
        errors="ignore",
    )
    X = df.drop(columns=["pct_cambio_7d"])
    y = df["pct_cambio_7d"].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# Función para dividir los datos en train, val y test
def dividir_datos_temporalmente(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.8,
        val_size: float = 0.1
):
    """
    Divide los datos temporalmente en conjuntos de entrenamiento, validación y prueba.

    :param X: Matriz de características escaladas.
    :param y: Serie con la variable objetivo.
    :param train_size: Proporción de datos para entrenamiento (por defecto 0.8).
    :param val_size: Proporción de datos para validación (por defecto 0.1).

    :returns: Tupla con los conjuntos de datos divididos:
        - X_train: Características de entrenamiento.
        - X_val: Características de validación.
        - X_test: Características de prueba.
        - y_train: Objetivo de entrenamiento.
        - y_val: Objetivo de validación.
        - y_test: Objetivo de prueba.
    """
    n = len(X)

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluar_modelo(
        modelo: SVR,
        X_test: np.ndarray,
        y_test: np.ndarray
):
    """
    Evalúa el rendimiento del modelo en el conjunto de prueba.

    :param modelo: Modelo SVR entrenado.
    :param X_test: Características del conjunto de prueba.
    :param y_test: Valores reales del objetivo en el conjunto de prueba.

    :returns: Tupla con las métricas de evaluación:
        - mae: Error absoluto medio.
        - mse: Error cuadrático medio.
        - rmse: Raíz del error cuadrático medio.
        - r2: Coeficiente de determinación R².
    """
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Accuracy direccional (comparar signo de la predicción y del valor real)
    pred_sign = np.sign(y_pred)
    real_sign = np.sign(y_test)
    directional_acc = accuracy_score(real_sign, pred_sign)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy Direccional: {directional_acc:.4f}")

    return mae, mse, rmse, r2, directional_acc

def predict_and_plot(
    model: SVR,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """
    Realiza predicciones y las grafica junto con los valores reales.

    :param model: Modelo entrenado.
    :param X_test: Datos de prueba.
    :param y_test: Etiquetas de prueba (sin escalar).
    """
    # Realizar predicciones
    predictions = model.predict(X_test).flatten()  # Asegurar que sea un array 1D

    # Graficar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Real", color="blue")
    plt.plot(predictions, label="Predicción", color="red", linestyle="dashed")
    plt.legend()
    plt.xlabel("Días")
    plt.ylabel("Valor de pct_cambio_7d")
    plt.title("Predicción vs Real: pct_cambio_7d")
    plt.show()


def entrenar_svr(
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1
):
    """
    Crea y entrena un modelo SVR con los parámetros especificados.

    :param X_train: Características de entrenamiento.
    :param y_train: Objetivo de entrenamiento.
    :param kernel: Tipo de kernel a utilizar ('linear', 'poly', 'rbf', 'sigmoid').
    :param C: Parámetro de regularización.
    :param epsilon: Margen de tolerancia en la función de pérdida.
    :return: Modelo SVR entrenado.
    """
    # Crear y entrenar el modelo SVR con los parámetros dados
    modelo = SVR(kernel=kernel, C=C, epsilon=epsilon)
    modelo.fit(X_train, y_train)

    return modelo

# Función para buscar los mejores hiperparámetros para SVR
def optimizar_hiperparametros(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: dict = None
):
    """
    Realiza una búsqueda de hiperparámetros para el modelo SVR probando diferentes combinaciones
    y evaluando cada modelo con el conjunto de validación.

    :param X_train: Características de entrenamiento.
    :param y_train: Objetivo de entrenamiento.
    :param X_val: Características de validación.
    :param y_val: Objetivo de validación.
    :param param_grid: Diccionario con los parámetros a probar. Si es None, se usa un grid predeterminado.
    :returns:
        - mejores_parametros: Diccionario con los mejores parámetros encontrados.
    """
    # Definir el grid de parámetros si no se proporciona uno
    if param_grid is None:
        param_grid = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 0.5, 1],
        }

    # Variables para almacenar el mejor modelo y score
    mejor_score = float("inf")
    mejores_parametros = None

    # Generar todas las combinaciones de parámetros

    param_combinations = list(product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    # Probar cada combinación de parámetros
    for params in param_combinations:
        # Crear diccionario de parámetros para esta iteración
        param_dict = dict(zip(param_keys, params))
        print(f"Probando configuración: {param_dict}")

        # Entrenar el modelo con los parámetros actuales
        modelo = entrenar_svr(
            X_train,
            y_train,
            kernel=param_dict['kernel'],
            C=param_dict['C'],
            epsilon=param_dict['epsilon']
        )

        # Evaluar el modelo en el conjunto de validación
        _, val_mse, _, _ = evaluar_modelo(modelo, X_val, y_val)

        # Verificar si este modelo es mejor que el anterior
        if val_mse < mejor_score:
            mejor_score = val_mse
            mejores_parametros = param_dict
            print("  ¡Nuevo mejor modelo encontrado!")

        print('\n')

    # Imprimir los mejores parámetros encontrados
    print("\nMejores hiperparámetros encontrados:")
    print(mejores_parametros)
    print(f"Mejor MSE en validación: {mejor_score:.4f}")

    return mejores_parametros

# Función principal
def main():
    mlflow.set_tracking_uri(r"file:../../mlruns")
    mlflow.set_experiment("SVR Optimization 7d")

    # Definir hiperparámetros a probar
    param_grid = {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5, 1],
    }

    # Preparar datos
    ruta_parquet = r"../../../../data/parquet/data_7d.parquet"

    fijar_semillas()
    df = cargar_datos(ruta_parquet)

    X, y, scaler = preprocesar_datos(df)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos_temporalmente(X, y)

    # Optimizar hiperparmetros
    mejores_parametros = optimizar_hiperparametros(X_train, y_train, X_val, y_val, param_grid)

    # Entrenar el modelo final con los mejores parámetros
    with mlflow.start_run(run_name="SVR"):
        mejor_modelo = entrenar_svr(
            X_train, y_train,
            kernel=mejores_parametros['kernel'],
            C=mejores_parametros['C'],
            epsilon=mejores_parametros['epsilon']
        )

        # Evaluar el modelo final en el conjunto de test
        mae, mse, rmse, r2, directional_acc = evaluar_modelo(mejor_modelo, X_test, y_test)

        # Registrar en MLflow
        mlflow.log_params(mejores_parametros)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "directional_acc": directional_acc})

    predict_and_plot(mejor_modelo, X_test, y_test)

if __name__ == "__main__":
    main()
