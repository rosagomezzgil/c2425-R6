import pandas as pd
import numpy as np

def calcular_rsi(df, window=14):
    delta = df["close_binance"].diff(1)
    ganancia = np.where(delta > 0, delta, 0)
    perdida = np.where(delta < 0, -delta, 0)

    media_ganancia = pd.Series(ganancia).rolling(window=window).mean()
    media_perdida = pd.Series(perdida).rolling(window=window).mean()

    rs = media_ganancia / media_perdida
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi
    return df

if __name__ == '__main__':
    df = pd.read_csv(r"../../data/datos_bitcoin_limpiados_binance.csv")
    df["var_diaria"] = df["close_binance"].pct_change() * 100
    df["volatilidad_50"] = df["close_binance"].rolling(window=50).std()
    df["volatilidad_200"] = df["close_binance"].rolling(window=200).std()
    df["rango_dia"] = df["high_binance"] - df["low_binance"]
    df["momentum_50"] = df["close_binance"] - df["close_binance"].shift(50)
    df["momentum_200"] = df["close_binance"] - df["close_binance"].shift(200)

    df = calcular_rsi(df)

    # Guardar el conjunto de datos
    output_path = r"../../data/creacion_variables_binance.csv"
    df.to_csv(output_path, encoding="utf-8-sig", index=False)