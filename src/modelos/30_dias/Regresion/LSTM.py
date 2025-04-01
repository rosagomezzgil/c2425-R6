import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.random import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.tensorflow
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# Fijar semillas
def fijar_semillas(seed: int = 123):
    """
    Fija las semillas para reproducibilidad

    :param seed: Semilla para las bibliotecas random, numpy y tensorflow
    """
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# 1. Cargar los datos
def load_data(parquet_file: str):
    """
    Carga los datos de un fichero parquet a un DataFrame de pandas

    :param parquet_file: URL del fichero parquet

    :return: pd.DataFrame
    """
    df = pd.read_parquet(parquet_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Asegurar que la fecha esté en formato datetime
    df.set_index('timestamp', inplace=True)
    return df

# 2. Preprocesar los datos
def preprocess_data(df: pd.DataFrame, horizonte: int, target_col: str, sequence_length: int = 30):
    """
    Preprocesa los datos para LSTM (normalización de X con StandardScaler y creación de secuencias)

    :param df: pd.DataFrame con los datos para el modelo
    :param horizonte: días de la predicción
    :param target_col: nombre de la columna de la variable objetivo
    :param sequence_length: longitud de las secuencias a crear (valor por defecto: 30)

    :return: X, y, scaler
    """
    assert target_col in df.columns, f"La columna {target_col} no está en el DataFrame"
    assert target_col == f'pct_cambio_{horizonte}d', "El nombre de la columna objetivo no coincide con el horizonte"

    col_categorica = f'tendencia_futura_{horizonte}d'
    assert col_categorica in df.columns, f"La columna {col_categorica} no está en el DataFrame"

    # Separar X y y antes de escalar
    features = df.drop(columns=[target_col, col_categorica])
    y = df[target_col].values  # Mantener y en su escala original

    # Escalar solo las características de entrada con StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

    X = []
    for i in range(len(features_scaled) - sequence_length - horizonte + 1):
        X.append(features_scaled.iloc[i: i + sequence_length].values)

    X = np.array(X)  # Convertir a numpy array

    return X, y[: len(X)], scaler  # Asegurar que y tenga la misma longitud que X


def split_data(
        X: np.array,
        y: np.array,
        test_size: float = 0.1,
        val_size: float = 0.1,
        shuffle: bool = False
):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.

    :param X: Características
    :param y: Etiquetas
    :param test_size: Proporción de datos para prueba
    :param val_size: Proporción de datos para validación (sobre el total)
    :param shuffle: Si se deben mezclar los datos antes de dividir
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Primero, dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)

    # Luego, dividir el conjunto de entrenamiento para extraer validación
    val_relative_size = val_size / (1 - test_size)  # Ajuste relativo al tamaño de entrenamiento
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_relative_size, shuffle=shuffle)

    return X_train, X_val, X_test, y_train, y_val, y_test


# 3. Crear y entrenar el modelo LSTM
def train_lstm(
    X_train: np.array,
    y_train: np.array,
    units: int = 50,
    epochs: int = 50,
    batch_size: int = 32,
):
    """
    Crea y entrena un modelo LSTM

    :param X_train: Datos de entrada para el entrenamiento
    :param y_train: Datos para el entrenamiento
    :param units: Número de unidades LSTM
    :param epochs: Número de épocas de entrenamiento
    :param batch_size: Tamaño del batch de entrenamiento

    :return: Modelo entrenado
    """

    print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
    print("Dispositivo en uso:", tf.test.gpu_device_name())

    model = Sequential(
        [
            LSTM(
                units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])
            ),
            Dropout(0.2),
            LSTM(units),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1
    )

    return model

def predict_and_plot(
    model: Sequential,
    X_test: np.array,
    y_test: np.array,
    target_col: str,
):
    """
    Realiza predicciones y las grafica junto con los valores reales.

    :param model: Modelo entrenado.
    :param X_test: Datos de prueba.
    :param y_test: Etiquetas de prueba (sin escalar).
    :param target_col: Columna objetivo a predecir.
    """
    # Realizar predicciones
    predictions = model.predict(X_test).flatten()  # Asegurar que sea un array 1D

    # Graficar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Real", color="blue")
    plt.plot(predictions, label="Predicción", color="red", linestyle="dashed")
    plt.legend()
    plt.xlabel("Días")
    plt.ylabel(f"Valor de {target_col}")
    plt.title(f"Predicción vs Real: {target_col}")
    plt.show()

def evaluate_model(model, X_test, y_test):
    """
    Calcula y muestra las métricas de evaluación del modelo.
    """
    predictions = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nMétricas del modelo:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    return mse, mae, rmse, r2


def optimize_hyperparameters(X_train, y_train, X_val, y_val, param_grid):
    """
    Realiza una búsqueda de hiperparámetros evaluando diferentes combinaciones.
    """
    best_score = float("inf")
    best_params = None

    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Probando configuración: {param_dict}")

        model = train_lstm(
            X_train, y_train,
            units=param_dict['units'],
            epochs=param_dict['epochs'],
            batch_size=param_dict['batch_size']
        )

        mse, _, _, _ = evaluate_model(model, X_val, y_val)

        if mse < best_score:
            best_score = mse
            best_params = param_dict
            print("Nuevo mejor modelo encontrado!")

        print('\n')

    print("\nMejores hiperparámetros encontrados:")
    print(best_params)
    return best_params

# 5. Ejecutar el flujo
def main():
    # Configurar MLflow en local
    mlflow.set_tracking_uri(r"file:../../mlruns")
    mlflow.set_experiment("LSTM Optimization 30d")

    # 1. Definir hiperparámetros a probar
    param_grid = {
        "units": [32, 64, 128],
        "epochs": [30, 50, 100, 500],
        "batch_size": [16, 32, 64],
    }

    # 2. Preparar datos
    parquet_file = "../../../../data/parquet/data_30d.parquet"
    horizonte = 30
    target_col = "pct_cambio_30d"
    # sequences_length = [1, 7, 30, 60]
    sequence_length = 7

    fijar_semillas()
    df = load_data(parquet_file)

    X, y, scaler = preprocess_data(df, horizonte, target_col, sequence_length)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 0.1, 0.1, shuffle=False)
    print(f"Datos preparados para sequence_length: {sequence_length}\n")

    # 3. Optimizar hiperparámetros
    best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, param_grid)

    # 4. Entrenar el modelo final con los mejores parámetros
    with mlflow.start_run(run_name=f"LSTM_seq{sequence_length}"):
        best_model = train_lstm(
            X_train, y_train,
            units=best_params['units'],
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size']
        )

        # 5. Evaluar
        mse, mae, rmse, r2 = evaluate_model(best_model, X_test, y_test)

        # Registrar en MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2})

    predict_and_plot(best_model, X_test, y_test, target_col)

if __name__ == "__main__":
    main()