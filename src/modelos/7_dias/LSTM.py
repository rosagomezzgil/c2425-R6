import mlflow
import numpy as np
import pandas as pd
from datetime import datetime

from src.utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d
from src.utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_7d
from src.utils.modelos_sklearn.C_seleccion_variables import seleccionar_variables
from src.utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_7d
from src.utils.modelos_sklearn.E_ajuste_hiperparametros import ajustar_modelo
from src.utils.modelos_sklearn.F_metricas import evaluar_modelo_7d
from src.utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from src.utils.mlflow_utils import setup_mlflow
from src.utils.paths import get_path

from src.modelos.arquitecturas.lstm_model import SklearnLikeLSTM  # Clase adaptada


def crear_ventanas(X, y, window_size=10):
    Xs, ys, fechas = [], [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y.iloc[i + window_size])
        fechas.append(y.index[i + window_size])
    return np.array(Xs), np.array(ys), pd.to_datetime(fechas)


def preparar_datos(window_size=10):
    df = cargar_datos_7d(get_path('data', 'parquet', 'data_7d.parquet'))
    X, y, scaler = preprocesar_datos_7d(df, tecnica="lstm", target_col="pct_cambio_7d")
    X, selector = seleccionar_variables(X, y, tecnica="lstm")
    X_seq, y_seq, fechas = crear_ventanas(X, y, window_size)

    test_size = 0.1

    X_train, y_train, X_test, y_test = dividir_datos_temporalmente_7d(X_seq, y_seq, test_size)

    n = len(fechas)
    test_start = int(n * (1 - test_size))
    fechas_train, fechas_test = fechas[:test_start], fechas[test_start:]

    return X_train, y_train, X_test, y_test, fechas_train, fechas_test



def main():
    setup_mlflow("LSTM-PyTorch")

    final_model = None
    final_params = None
    final_window_size = None
    data = {}
    best_mse = np.inf

    windows_size = [1, 7, 14, 21, 28, 30, 35]

    for window_size in windows_size:
        X_train, y_train, X_test, y_test, fechas_train, fechas_test = preparar_datos(window_size)
        input_shape = X_train.shape[1:]  # (seq_len, n_features)

        lstm_modelo = SklearnLikeLSTM()

        param_grid = {
            "hidden_size": [32, 64, 128],
            "num_layers": [1, 2],
            "dropout": [0.0, 0.2, 0.5],
            "epochs": [20, 50],
            "batch_size": [16, 32, 64],
            "lr": [0.001, 0.0005],
            "input_shape": [input_shape],  # necesario para instanciar el modelo correctamente
        }

        hora_actual = datetime.now().strftime("%H-%M-%S")
        with mlflow.start_run(run_name=f"LSTM-SklearnLike-window_size {window_size}-{hora_actual}"):
            best_model, best_params = ajustar_modelo(lstm_modelo, X_train, y_train, param_grid, strategy="grid")

            # Evaluación del modelo
            mse, mae, rmse, r2, acc = evaluar_modelo_7d(best_model, X_test, y_test)

            fig = plot_predictions(y_test, modelo=best_model, X=X_test, fechas=fechas_test, show=False)

            # Loguear los mejores parámetros y métricas
            mlflow.log_params(best_params)
            mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})
            mlflow.log_figure(fig, "prediction_plot.png")

            if mse < best_mse:
                best_mse = mse
                final_model = best_model
                final_params = best_params
                final_window_size = window_size

                data = {
                    "X_test": X_test,
                    "y_test": y_test,
                    "fechas_test": fechas_test
                }

    hora_actual = datetime.now().strftime("%H-%M-%S")
    with mlflow.start_run(run_name=f"BEST-LSTM-SklearnLike-window_size {final_window_size}-{hora_actual}"):
        print("\nMejor modelo encontrado:")
        mse, mae, rmse, r2, acc = evaluar_modelo_7d(final_model, data['X_test'], data['y_test'])

        fig = plot_predictions(data['y_test'], modelo=final_model, X=data['X_test'], fechas=data['fechas_test'])

        # Loguear los mejores parámetros y métricas
        mlflow.log_params(final_params)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})
        mlflow.log_figure(fig, "prediction_plot.png")



if __name__ == "__main__":
    main()
