from Calcular_RSI import calcular_rsi
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv(r"../../data/datos_bitcoin_limpiados_binance.csv")
    df["var_diaria"] = df["close_binance"].pct_change() * 100
    df["volatilidad_50"] = df["close_binance"].rolling(window=50).std()
    df["volatilidad_200"] = df["close_binance"].rolling(window=200).std()
    df["rango_dia"] = df["high_binance"] - df["low_binance"]
    df["momentum_50"] = df["close_binance"] - df["close_binance"].shift(50)
    df["momentum_200"] = df["close_binance"] - df["close_binance"].shift(200)

    df = calcular_rsi(df, 'close_binance')

    # Guardar el conjunto de datos
    output_path = r"../../data/creacion_variables_binance.csv"
    df.to_csv(output_path, encoding="utf-8-sig", index=False)