from Calcular_RSI import calcular_rsi
import pandas as pd

if __name__ == '__main__':
    # Cargar datos desde CSV
    sp500 = pd.read_csv(r"../../data/extraccion/datos_sp500.csv")

    # Ver si hay valores nulos
    print(sp500.isnull().sum())  # No tiene valores nulos

    # Ver el tipo de datos
    print(sp500.info())

    # Convertir timestamp a formato fecha
    sp500['timestamp'] = pd.to_datetime(sp500['timestamp'])
    sp500.sort_values('timestamp', inplace=True)  # Ordenar por fecha

    # Convertir todas las columnas excepto 'timestamp' a numéricas
    for col in sp500.columns:
        if col != "timestamp":
            sp500[col] = pd.to_numeric(sp500[col], errors="coerce")

    # Agregar variables de análisis técnico
    # Medias Móviles
    sp500['SMA_50'] = sp500['Close'].rolling(window=50).mean()
    sp500['SMA_200'] = sp500['Close'].rolling(window=200).mean()
    sp500['EMA_50'] = sp500['Close'].ewm(span=50, adjust=False).mean()
    sp500['EMA_200'] = sp500['Close'].ewm(span=200, adjust=False).mean()

    # Volatilidad (Desviación estándar de los últimos X días)
    sp500["volatilidad_50"] = sp500["Close"].rolling(window=50).std()
    sp500["volatilidad_200"] = sp500["Close"].rolling(window=200).std()

    # Momentum (Diferencia entre el precio actual y hace X días)
    sp500["momentum_50"] = sp500["Close"] - sp500["Close"].shift(50)
    sp500["momentum_200"] = sp500["Close"] - sp500["Close"].shift(200)

    # Variación diaria en porcentaje
    sp500["var_diaria"] = sp500["Close"].pct_change() * 100

    # Rango Diario (Diferencia entre el precio más alto y más bajo)
    sp500["rango_dia"] = sp500["High"] - sp500["Low"]

    # RSI (Relative Strength Index)
    sp500 = calcular_rsi(sp500, 'Close')

    # Guardar el conjunto de datos procesados
    output_path = r"../../data/limpieza/limpios_sp500.csv"
    sp500.to_csv(output_path, index=False)

    print(f"Archivo guardado en: {output_path}")
    print(sp500.head())  # Mostrar primeras filas para verificación
