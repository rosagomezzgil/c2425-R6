from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.random import set_seed
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
        sequence_length: int = 30):
    """
    Preprocesa los datos para LSTM (normalización y creación de secuencias)

    :param df: pd.DataFrame con los datos para el modelo
    :param horizonte: días de la predicción
    :param target_col: nombre de la columna de la variable objetivo
    :param sequence_length: longitud de las secuencias a crear (valor por defecto: 30)

    :return: X, y, scaler
    """
    # Escalado de los datos
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    assert target_col in df.columns

    X, y = [], []
    for i in range(len(df_scaled) - sequence_length - horizonte + 1):  # Ajustar el rango para las secuencias
        X.append(df_scaled[i:i + sequence_length])  # Se usa sequence_length para definir la secuencia
        y.append(df_scaled[i + sequence_length + horizonte - 1][df.columns.get_loc(target_col)])

    # Convertir listas a arrays de numpy
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

def predict_and_plot(
        model: Sequential,
        X_test: np.array,
        y_test: np.array,
        scaler: MinMaxScaler,
        target_col: str):
    """
    Realiza predicciones y las grafica junto con los valores reales.

    :param model: Modelo entrenado.
    :param X_test: Datos de prueba.
    :param y_test: Etiquetas de prueba.
    :param scaler: Escalador para desescalar los datos.
    :param target_col: Columna objetivo a predecir.
    """
    # Realizar predicciones
    predictions = model.predict(X_test)

    # Desescalar las predicciones
    predictions = scaler.inverse_transform(
        np.concatenate([X_test[:, -1, :-1], predictions], axis=-1)
    )[:, -1]  # Sólo tomamos la columna correspondiente al target (target_col)

    # Desescalar los valores reales (y_test)
    y_test_rescaled = scaler.inverse_transform(
        np.concatenate([X_test[:, -1, :-1], y_test.reshape(-1, 1)], axis=-1)
    )[:, -1]  # Sólo tomamos la columna correspondiente al target (target_col)

    # Graficar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label='Real')
    plt.plot(predictions, label='Predicción')
    plt.legend()
    plt.xlabel('Días')
    plt.ylabel(f'{target_col} Predicción')
    plt.title(f'Predicción vs Real: {target_col}')
    plt.show()


# 5. Ejecutar el flujo
def main(
        parquet_file: str,
        horizonte: int = 7,
        target_col: str = '', # TODO
        units: int = 50,
        epochs: int = 50,
        batch_size: int = 32):
    """
    Ejecuta el flujo completo del modelo LSTM

    :param parquet_file: Ruta del archivo parquet
    :param horizonte: Número de días a predecir
    :param target_col: Columna de la variable objetivo
    :param units: Número de unidades LSTM
    :param epochs: Número de épocas de entrenamiento
    :param batch_size: Tamaño del batch
    """
    fijar_semillas()
    df = load_data(parquet_file)

    X, y, scaler = preprocess_data(df, horizonte, target_col)

    # shuffle=False --> asegura que los datos se dividan en orden cronológico
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = train_lstm(X_train, y_train, horizonte, units, epochs, batch_size)
    predict_and_plot(model, X_test, y_test, scaler, target_col)


if __name__ == "__main__":
    # Parámetros para ejecutar el modelo
    parquet_file = ''  # TODO
    horizonte = 7  # Número de días a predecir
    target_col = 'bitcoin_change_7d'  # Columna objetivo, porcentaje de cambio en 7 días
    main(parquet_file, horizonte, target_col)
