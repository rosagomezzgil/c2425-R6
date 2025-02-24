import pandas as pd
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv(r"../../data/datos_coingecko.csv")

    print(df.info())  # Ver tipos de datos y valores nulos
    print(df.describe())  # Estadísticas básicas
    print(df.head())  # Primeras filas

    #convertimos la columna timestamp a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date

    df["var_diaria"] = df["price"].pct_change() * 100 # % de variación
    df["volatilidad_50"] = df["price"].rolling(window=50).std()
    df["volatilidad_200"] = df["price"].rolling(window=200).std()
    df["rango_dia"] = df["price"].diff().abs()
    df["momentum_50"] = df["price"] - df["price"].shift(50)
    df["momentum_200"] = df["price"] - df["price"].shift(200)
    # Media Móvil Simple (SMA)
    df['SMA_50'] = df['price'].rolling(window=50).mean()  # 50 días
    df['SMA_200'] = df['price'].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df['EMA_50'] = df['price'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['price'].ewm(span=200, adjust=False).mean()


    def calcular_rsi(df, window=14):
        delta = df["price"].diff(1)
        ganancia = np.where(delta > 0, delta, 0)
        perdida = np.where(delta < 0, -delta, 0)

        media_ganancia = pd.Series(ganancia).rolling(window=window).mean()
        media_perdida = pd.Series(perdida).rolling(window=window).mean()

        rs = media_ganancia / media_perdida
        rsi = 100 - (100 / (1 + rs))

        df["RSI"] = rsi
        return df


    df = calcular_rsi(df)


    output_path = r"../../data/datos_coingecko_limpios.csv"
    df.to_csv(output_path, encoding = "utf-8-sig", index=False)
    print(df)





