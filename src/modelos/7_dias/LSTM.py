import mlflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from src.modelos.A_cargar_datos import cargar_datos_7d
from src.modelos.B_preprocesamiento import preprocesar_datos_7d
from src.modelos.E_ajuste_hiperparametros import ajustar_modelo
from src.modelos.F_metricas import evaluar_modelo_7d

# Genera ventanas para LSTM
def crear_ventanas(X, y, window_size=10):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)


def preparar_datos():
    # Cargar datos
    df = cargar_datos_7d("../../../data/parquet/data_7d.parquet")

    # Preprocesar datos para LSTM
    X, y, scaler = preprocesar_datos_7d(df, tecnica="lstm", target_col="pct_cambio_7d")

    # Crear secuencias para LSTM
    X_seq, y_seq = crear_ventanas(X, y)

    # Dividir en entrenamiento y test
    n_train = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:n_train], X_seq[n_train:]
    y_train, y_test = y_seq[:n_train], y_seq[n_train:]

    return X_train, y_train, X_test, y_test


# Crear el modelo LSTM
def crear_modelo_lstm(input_shape, units=50, epochs=20, batch_size=32):
    model = Sequential([
        LSTM(units, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def main():
    mlflow.set_tracking_uri("file:../../mlruns")
    mlflow.set_experiment("Modelo LSTM con TimeSeriesSplit")

    # Preparar los datos
    X_train, y_train, X_test, y_test = preparar_datos()

    # Ajuste de hiperparámetros: para LSTM ajustamos epochs, batch_size y unidades LSTM
    param_grid = {
        "units": [50, 100],
        "epochs": [10, 20],
        "batch_size": [16, 32]
    }

    # Utilizamos GridSearch o RandomizedSearch con TimeSeriesSplit
    best_model, best_params = ajustar_modelo(crear_modelo_lstm, X_train, y_train, param_grid, strategy="grid")

    # Evaluar el modelo final con el mejor conjunto de hiperparámetros
    with mlflow.start_run():
        mse, mae, rmse, r2, acc = evaluar_modelo_7d(best_model, X_test, y_test)

        # Loguear el modelo y los parámetros en MLFlow
        mlflow.log_params(best_params)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})


if __name__ == "__main__":
    main()
