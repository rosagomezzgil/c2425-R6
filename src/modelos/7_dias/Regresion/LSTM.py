from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.random import set_seed
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
    df['date'] = pd.to_datetime(df['date'])  # Asegurar que la fecha esté en formato datetime
    df.set_index('date', inplace=True)
    return df

# 2. Preprocesar los datos
def preprocess_data(
        df: pd.DataFrame,
        horizonte: int,
        target_col: str,
        sequence_length: int):
    """
    Preprocesa los datos para LSTM (normalización y creación de secuencias)

    :param df: pd.DataFrame con los datos para el modelo
    :param horizonte: días de la predicción
    :param target_col: nombre de la columna de la variable objetivo
    :param sequence_length: longitud de las secuencias a crear

    :return: X, y, scaler
    """
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df_scaled) - horizonte):
        X.append(df_scaled[i:i + horizonte])
        y.append(df_scaled[i + horizonte][df.columns.get_loc(target_col)])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# 3. Crear y entrenar el modelo LSTM
def train_lstm(
        X_train: np.array,
        y_train: np.array,
        horizonte: int,
        units: int = 50,
        epochs: int = 50,
        batch_size: int = 32):
    """
    Crea y entrena un modelo LSTM

    :param X_train: Datos de entrada para el entrenamiento
    :param y_train: Datos para el entrenamiento
    :param horizonte: Días de predicción
    :param units: Número de unidades LSTM
    :param epochs: Número de épocas de entrenamiento
    :param batch_size: Tamaño del batch de entrenamiento

    :return: Modelo entrenado
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(horizonte, X_train.shape[2])),
        Dropout(0.2),
        LSTM(units),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    return model


# 4. Hacer predicciones y graficar
def predict_and_plot(
        model: Sequential,
        X_test: np.array,
        y_test: np.array,
        scaler: MinMaxScaler,
        target_col: str):
    """
    Realiza predicciones y las grafica junto con los valores reales

    :param model: Modelo entrenado
    :param X_test: Datos de prueba
    :param y_test: Etiquetas de prueba
    :param scaler: Escalador para desescalar los datos
    :param target_col: Columna objetivo a predecir
    """
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate([X_test[:, -1, :-1], predictions], axis=-1))[:, -1]  # Desescalar
    y_test = scaler.inverse_transform(np.concatenate([X_test[:, -1, :-1], y_test.reshape(-1, 1)], axis=-1))[:, -1]

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Real')
    plt.plot(predictions, label='Predicción')
    plt.legend()
    plt.show()


# 5. Ejecutar el flujo
def main(
        parquet_file: str,
        horizonte: int,
        target_col: str,
        columns: list,
        units: int = 50,
        epochs: int = 50,
        batch_size: int = 32):
    """
    Ejecuta el flujo completo del modelo LSTM

    :param parquet_file: Ruta del archivo parquet
    :param horizonte: Número de días a predecir
    :param target_col: Columna de la variable objetivo
    :param columns: Lista de columnas a usar
    :param units: Número de unidades LSTM
    :param epochs: Número de épocas de entrenamiento
    :param batch_size: Tamaño del batch
    """
    fijar_semillas()
    df = load_data(parquet_file)

    X, y, scaler = preprocess_data(df, horizonte, target_col, sequence_length=len(columns) - 1)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = train_lstm(X_train, y_train, horizonte, units, epochs, batch_size)
    predict_and_plot(model, X_test, y_test, scaler, target_col)


if __name__ == "__main__":
    # Parámetros para ejecutar el modelo
    parquet_file = 'bitcoin_data.parquet'  # Asegúrate de tener este archivo
    horizonte = 7  # Número de días a predecir
    target_col = 'bitcoin_change_7d'  # Columna objetivo, porcentaje de cambio en 7 días
    columns = ['bitcoin_change_7d', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9']  # Columnas de entrada
    main(parquet_file, horizonte, target_col, columns, units=50, epochs=50, batch_size=32)
