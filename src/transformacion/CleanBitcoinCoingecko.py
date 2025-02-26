from Calcular_RSI import calcular_rsi
import pandas as pd

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

    df = calcular_rsi(df, 'price')

    output_path = r"../../data/datos_coingecko_limpios.csv"
    df.to_csv(output_path, encoding = "utf-8-sig", index=False)
    print(df)





