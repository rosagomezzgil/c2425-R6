"""
    Este script limpia y transforma los datos históricos de precios de Bitcoin provenientes
    de la plataforma CoinGecko. Comienza convirtiendo el timestamp a formato de fecha y
    renombrando la columna "price" como "close" para unificar los nombres con otros datasets.

    Luego calcula algunos indicadores técnicos como variación diaria, volatilidad, rango diario,
    momentum, medias móviles (SMA y EMA) y el RSI llamando a la función definida. Estos indicadores
    enriquecen el dataset y lo preparan para análisis predictivos.

    Finalmente, el archivo limpio se guarda como CSV listo para su uso
    en las siguientes etapas del proyecto.

    """


from Calcular_RSI import calcular_rsi
from src.utils.paths import  get_path
import pandas as pd
import os

if __name__ == "__main__":
    df = pd.read_csv(get_path('data', 'extraccion', 'datos_coingecko.csv'))

    print(df.info())  # Ver tipos de datos y valores nulos
    print(df.describe())  # Estadísticas básicas
    print(df.head())  # Primeras filas

    # convertimos la columna timestamp a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date
    df = df.rename(
        columns={"price": "close"}
    )  # se entiende como precio de cierre para el estudio

    df["var_diaria"] = df["close"].pct_change() * 100  # % de variación
    df["volatilidad_50"] = df["close"].rolling(window=50).std()
    df["volatilidad_200"] = df["close"].rolling(window=200).std()
    df["rango_dia"] = df["close"].diff().abs()
    df["momentum_50"] = df["close"] - df["close"].shift(50)
    df["momentum_200"] = df["close"] - df["close"].shift(200)
    # Media Móvil Simple (SMA)
    df["SMA_50"] = df["close"].rolling(window=50).mean()  # 50 días
    df["SMA_200"] = df["close"].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()

    df = calcular_rsi(df, "close")

    output_path = get_path('data', 'limpieza', 'limpios_coingecko.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df)
