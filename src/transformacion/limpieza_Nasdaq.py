
import pandas as pd
import numpy as np

def calcular_rsi(df, window=14):
    """
    Calcula el RSI (Relative Strength Index) con una ventana determinada.
    """
    delta = df["Close"].diff(1)
    ganancia = np.where(delta > 0, delta, 0)
    perdida = np.where(delta < 0, -delta, 0)

    media_ganancia = pd.Series(ganancia).rolling(window=window).mean()
    media_perdida = pd.Series(perdida).rolling(window=window).mean()

    rs = media_ganancia / media_perdida
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi
    return df


if __name__ == '__main__':
    # Cargar datos desde CSV
    nasdaq = pd.read_csv(r"../../data/datos_bitcoin_nasdaq.csv")

    # Ver si hay valores nulos
    print(nasdaq.isnull().sum())    #No tiene valores nulos

    #Ver el tipo de datos
    print(nasdaq.info())

    # Convertir timestamp a formato fecha
    nasdaq['timestamp'] = pd.to_datetime(nasdaq['timestamp'])
    nasdaq.sort_values('timestamp', inplace=True)  # Ordenar por fecha

    # Convertir todas las columnas excepto 'timestamp' a numéricas
    for col in nasdaq.columns:
        if col != "timestamp":
            nasdaq[col] = pd.to_numeric(nasdaq[col], errors="coerce")

    #  Agregar variables de análisis técnico

    #  Medias Móviles
    nasdaq['SMA_50'] = nasdaq['Close'].rolling(window=50).mean()
    nasdaq['SMA_200'] = nasdaq['Close'].rolling(window=200).mean()
    nasdaq['EMA_50'] = nasdaq['Close'].ewm(span=50, adjust=False).mean()
    nasdaq['EMA_200'] = nasdaq['Close'].ewm(span=200, adjust=False).mean()

    #  Volatilidad (Desviación estándar de los últimos X días)
    nasdaq["volatilidad_50"] = nasdaq["Close"].rolling(window=50).std()
    nasdaq["volatilidad_200"] = nasdaq["Close"].rolling(window=200).std()

    #  Momentum (Diferencia entre el precio actual y hace X días)
    nasdaq["momentum_50"] = nasdaq["Close"] - nasdaq["Close"].shift(50)
    nasdaq["momentum_200"] = nasdaq["Close"] - nasdaq["Close"].shift(200)

    #  Variación diaria en porcentaje
    nasdaq["var_diaria"] = nasdaq["Close"].pct_change() * 100

    #  Rango Diario (Diferencia entre el precio más alto y más bajo)
    nasdaq["rango_dia"] = nasdaq["High"] - nasdaq["Low"]

    #  RSI (Relative Strength Index)
    nasdaq = calcular_rsi(nasdaq)

    # Guardar el conjunto de datos procesados
    output_path = r"../../data/datos_bitcoin_limpiados_nasdaq.csv"
    nasdaq.to_csv(output_path, index=False)

    print(f"Archivo guardado en: {output_path}")
    print(nasdaq.head())  # Mostrar primeras filas para verificación

