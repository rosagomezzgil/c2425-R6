import pandas as pd
import numpy as np
def calcular_rsi(df, window=14):
    delta = df["close"].diff(1)
    ganancia = np.where(delta > 0, delta, 0)
    perdida = np.where(delta < 0, -delta, 0)

    media_ganancia = pd.Series(ganancia).rolling(window=window).mean()
    media_perdida = pd.Series(perdida).rolling(window=window).mean()

    rs = media_ganancia / media_perdida
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi
    return df

if __name__ == '__main__':
    df = pd.read_csv(r"../../data/datos_mercadoDerivados_binance.csv")

    # Mostrar las primeras filas del DataFrame
    print("Primeras filas del archivo CSV:")
    print(df.head())

    # Mostrar información general del DataFrame
    print("\nInformación del dataset:")
    print(df.info())

    # Mostrar estadísticas básicas de los datos
    print("\nResumen estadístico:")
    print(df.describe())

    # Eliminar duplicados y nulos
    df = df.drop_duplicates()
    df = df.dropna()

    df.columns = df.columns.str.lower()

    # Mostrar resultado
    print("Limpieza de formato de datos:")
    print(df.head())
    print(df.info())

    ### DETECCIÓN Y TRATAMIENTO DE VALORES NULOS ###
    print("\nValores nulos en el dataset:")
    print(df.isnull().sum())


    ### DETECCIÓN DE OUTLIERS ###
    # No vamos a eliminar outliers ya que pueden ser utiles para la prediccion, solamente los detectamos

    ### RESULTADO FINAL ###
    print("\nDataset limpio y listo para su análisis:")
    print(df.head())

    df["var_diaria"] = df["close"].pct_change() * 100
    df["volatilidad_50"] = df["close"].rolling(window=50).std()
    df["volatilidad_200"] = df["close"].rolling(window=200).std()
    df["rango_dia"] = df["high"] - df["low"]
    df["momentum_50"] = df["close"] - df["close"].shift(50)
    df["momentum_200"] = df["close"] - df["close"].shift(200)

    df = calcular_rsi(df)

    df.to_csv("../../data/mercadoDerivados_binance_limpio.csv", encoding="utf-8-sig", index=False)
    print(df)
