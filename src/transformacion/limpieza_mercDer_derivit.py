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


    df = df.sort_values(by="timestamp")


    # Verificar resultado
    print(df_diario.head())

    df["var_diaria"] = df["close"].pct_change() * 100
    df["volatilidad_50"] = df["close"].rolling(window=50).std()
    df["volatilidad_200"] = df["close"].rolling(window=200).std()
    df["rango_dia"] = df["close"] - df["open"]
    df["momentum_50"] = df["close"] - df["close"].shift(50)
    df["momentum_200"] = df["close"] - df["close"].shift(200)
    # Media Móvil Simple (SMA)
    df['SMA_50'] = df['close'].rolling(window=50).mean()  # 50 días
    df['SMA_200'] = df['close'].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

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

    df = calcular_rsi(df)


    # Guardar el conjunto de datos
    output_path = r"../../data/limpieza_mercDer_deribit.csv"
    df.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df)
