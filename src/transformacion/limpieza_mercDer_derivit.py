import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Cargar el DataFrame
    df = pd.read_csv(r"../../data/datos_mercadoDerivados_derivit.csv", parse_dates=["timestamp"])

    # Convertir timestamp a datetime si aún no lo es
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Agrupar por día y calcular los valores relevantes
    df_diario = df.resample("D", on="timestamp").agg({
        "open": "first",    # Primer precio del día
        "high": "max",      # Precio máximo del día
        "low": "min",       # Precio mínimo del día
        "close": "last",    # Último precio del día
        "volume": "sum"     # Volumen total del día
    }).reset_index()


    df_diario = df_diario.sort_values(by="timestamp")


    # Verificar resultado
    print(df_diario.head())

    df_diario["var_diaria"] = df_diario["close"].pct_change() * 100
    df_diario["volatilidad_50"] = df_diario["close"].rolling(window=50).std()
    df_diario["volatilidad_200"] = df_diario["close"].rolling(window=200).std()
    df_diario["rango_dia"] = df_diario["close"] - df_diario["open"]
    df_diario["momentum_50"] = df_diario["close"] - df_diario["close"].shift(50)
    df_diario["momentum_200"] = df_diario["close"] - df_diario["close"].shift(200)
    # Media Móvil Simple (SMA)
    df_diario['SMA_50'] = df_diario['close'].rolling(window=50).mean()  # 50 días
    df_diario['SMA_200'] = df_diario['close'].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df_diario['EMA_50'] = df_diario['close'].ewm(span=50, adjust=False).mean()
    df_diario['EMA_200'] = df_diario['close'].ewm(span=200, adjust=False).mean()

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

    df_diario = calcular_rsi(df_diario)


    # Guardar el conjunto de datos
    output_path = r"../../data/limpieza_mercDer_deribit.csv"
    df_diario.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df_diario)
