import numpy as np
import pandas as pd

def calcular_rsi(df, columna: str, window: int = 14):
    """
    Calcula el Índice de Fuerza Relativa (RSI) para una serie de precios.

    :param df: DataFrame que contiene una columna "price" con los precios del activo.
    :param columna: El nombre de la columna con los precios para el cálculo.
    :param window: Ventana de tiempo para el cálculo del RSI (por defecto 14 días).
    :return: DataFrame con una nueva columna "RSI" que contiene los valores calculados.
    """
    # Calcular la diferencia de precios
    delta = df[columna].diff(1)

    # Calcular las ganancias y pérdidas
    ganancia = np.where(delta > 0, delta, 0)
    perdida = np.where(delta < 0, -delta, 0)

    # Calcular la media de las ganancias y pérdidas en la ventana especificada
    media_ganancia = pd.Series(ganancia).rolling(window=window).mean()
    media_perdida = pd.Series(perdida).rolling(window=window).mean()

    # Calcular el RS (relación de las ganancias y pérdidas promedio)
    rs = media_ganancia / media_perdida

    # Calcular el RSI
    rsi = 100 - (100 / (1 + rs))

    # Añadir la columna de RSI al DataFrame
    df["RSI"] = rsi
    return df
