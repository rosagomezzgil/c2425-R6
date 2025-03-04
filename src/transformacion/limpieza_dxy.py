import pandas as pd
from Calcular_RSI import calcular_rsi

if __name__ == '__main__':

    dxy = pd.read_csv(r"../../data/extraccion/datos_dxy.csv")

    #Hacemos un analisis inicial de como se comporta el df
    print(dxy.head())
    print(dxy.info())
    print(dxy.describe()) #Observamos que el volumen es 0

    # El DXY es solo un índice, no un activo negociable. Como no se compra ni se vende directamente, no tiene volumen de trading, por lo que eliminamos la variable pues no tiene informacion util
    dxy = dxy.drop(columns=['Volume'], errors='ignore')

    # Convertir timestamp a formato fecha
    dxy['timestamp'] = pd.to_datetime(dxy['timestamp'])
    dxy.sort_values('timestamp', inplace=True)  # Ordenar por fecha

    # Convertir todas las columnas excepto 'timestamp' a numéricas
    for col in dxy.columns:
        if col != "timestamp":
            dxy[col] = pd.to_numeric(dxy[col], errors="coerce")

    # Agregar variables de análisis técnico
    # Medias Móviles
    dxy['SMA_50'] = dxy['Close'].rolling(window=50).mean()
    dxy['SMA_200'] = dxy['Close'].rolling(window=200).mean()
    dxy['EMA_50'] = dxy['Close'].ewm(span=50, adjust=False).mean()
    dxy['EMA_200'] = dxy['Close'].ewm(span=200, adjust=False).mean()

    # Volatilidad (Desviación estándar de los últimos X días)
    dxy["volatilidad_50"] = dxy["Close"].rolling(window=50).std()
    dxy["volatilidad_200"] = dxy["Close"].rolling(window=200).std()

    # Momentum (Diferencia entre el precio actual y hace X días)
    dxy["momentum_50"] = dxy["Close"] - dxy["Close"].shift(50)
    dxy["momentum_200"] = dxy["Close"] - dxy["Close"].shift(200)

    # Variación diaria en porcentaje
    dxy["var_diaria"] = dxy["Close"].pct_change(fill_method=None) * 100

    # Rango Diario (Diferencia entre el precio más alto y más bajo)
    dxy["rango_dia"] = dxy["High"] - dxy["Low"]

    # RSI (Relative Strength Index)
    sp500 = calcular_rsi(dxy, 'Close')

    # Guardar el conjunto de datos procesados
    output_path = r"../../data/limpieza/limpios_dxy.csv"
    dxy.to_csv(output_path, index=False)

    print(f"Archivo guardado en: {output_path}")
    print(dxy.head())  # Mostrar primeras filas para verificación

    # Mostrar las ultimas filas para verificar cambios
    print(dxy.tail())
