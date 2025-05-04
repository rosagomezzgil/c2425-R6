"""
Módulo: VAR.py

Descripción:
Este script implementa un modelo VAR (Vector Autoregressive) para modelar y predecir series temporales multivariadas,
con foco en predecir la variación porcentual a 30 días (`pct_cambio_30d`). El modelo considera la influencia conjunta
de múltiples variables temporales en la predicción futura del target.

Características principales:
- Carga y preprocesamiento del dataset multivariado.
- Selección de variables relevantes mediante una técnica adaptada a 4 modelos VAR.
- División temporal de datos para garantizar la validez temporal.
- Validación cruzada específica para series temporales para determinar el mejor número de rezagos (lags).
- Entrenamiento del modelo VAR con el mejor `lag`.
- Predicción multivariada sobre el conjunto de prueba y evaluación específica del target.
- Registro del experimento, parámetros, métricas y visualizaciones con MLflow.

Dependencias:
- pandas
- numpy
- statsmodels
- scikit-learn
- matplotlib
- mlflow
- utils personalizados:
    - utils.modelos_sklearn.* (carga, preprocesamiento, división, selección de variables, métricas, gráficos)
    - utils.paths
    - utils.mlflow_utils

Uso:
Ejecutar este script para entrenar un modelo VAR sobre un conjunto de series temporales multivariadas,
ideal para analizar relaciones dinámicas entre variables financieras o económicas a lo largo del tiempo.

"""


from utils.modelos_sklearn.A_cargar_datos import cargar_datos_30d
from utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_30d
from utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_30d
from utils.modelos_sklearn.C_seleccion_variables import seleccionar_variables
from utils.modelos_sklearn.F_metricas import evaluar_modelo_30d
from utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from utils.mlflow_utils import setup_mlflow
from utils.paths import get_path
from statsmodels.tsa.api import VAR
import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from datetime import datetime


def main(parquet_path: str, target_col: str = 'pct_cambio_30d'):
    # Cargar datos y preprocesarlos
    df = cargar_datos_30d(parquet_path)
    df = preprocesar_datos_30d(df, target_col)  # Este paso ya te asegura que hay múltiples columnas

    # Definir X e y
    X = df.drop(columns=[target_col])  # Las variables explicativas
    y = df[target_col]  # La variable target

    # Seleccionar las variables para VAR
    X_var, _ = seleccionar_variables(X, y, tecnica="var")

    # Dividir los datos en conjunto de entrenamiento y test --> ponemos el multivariado a true porque VAR funciona
    # con vectores
    train_df, test_df = dividir_datos_temporalmente_30d(df, target_col, test_ratio=0.2, multivariado=True)


    setup_mlflow("Modelo VAR")

    tscv = TimeSeriesSplit(n_splits=5)
    errores_por_lag = {}

    for lag in range(1, 16):
        errores = []

        for train_index, val_index in tscv.split(df):
            train_split, val_split = df.iloc[train_index], df.iloc[val_index]

            try:
                modelo = VAR(train_split)
                fitted = modelo.fit(lag)
                forecast = fitted.forecast(train_split.values[-lag:], steps=len(val_split))

                mse = mean_squared_error(val_split, forecast)
                errores.append(mse)
            except:
                continue

        if errores:
            errores_por_lag[lag] = np.mean(errores)

    if not errores_por_lag:
        raise ValueError("Ningún modelo VAR fue válido con los datos y splits proporcionados.")

    best_lag = min(errores_por_lag, key=errores_por_lag.get)
    best_model = VAR(train_df).fit(best_lag)

    hora_actual = datetime.now().strftime("%H-%M-%S")
    with mlflow.start_run(run_name=f"VAR-{hora_actual}"):
        # Hacer predicciones sobre el test
        n_obs = len(test_df)
        forecast = best_model.forecast(train_df.values[-best_lag:], steps=n_obs)
        forecast_df = test_df.copy().astype(np.float64)
        forecast_df.iloc[:, :] = forecast

        # Evaluar solo la serie target
        y_true = test_df[target_col]
        y_pred = forecast_df[target_col]

        mse, mae, rmse, r2, acc = evaluar_modelo_30d(y_true, y_pred)

        fig = plot_predictions(y_true, pred=y_pred)

        mlflow.log_param("best_lag", best_lag)
        mlflow.log_metrics({
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "acc": acc
        })
        mlflow.log_figure(fig, "prediction_plot.png")


if __name__ == "__main__":
    main(get_path('data', 'parquet', 'data_30d.parquet'))

