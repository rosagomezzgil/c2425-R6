import pandas as pd
import numpy as np


def convertir_euros(valor):
    """
    Función para convertir valores monetarios en float
    """
    return float(valor.replace("€", "").replace(",", "")) if isinstance(valor, str) else valor


def convertir_volumen(vol):
    """
    Función para convertir volumen (B de Billones a número real)
    """
    if isinstance(vol, str) and 'B' in vol:  # Solo aplicar reemplazos si vol es un string
            return convertir_euros(vol.replace("B", "")) * 1e9  # Convertir a número real

    return convertir_euros(vol)


def detectar_outliers_iqr(data, col):
    """
    Definir una función para detectar outliers con el método IQR
    """
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = data[(data[col] < limite_inferior) | (data[col] > limite_superior)]
    return outliers


def calcular_rsi(df, window=14):
    delta = df["cerrar**"].diff(1)
    ganancia = np.where(delta > 0, delta, 0)
    perdida = np.where(delta < 0, -delta, 0)

    media_ganancia = pd.Series(ganancia).rolling(window=window).mean()
    media_perdida = pd.Series(perdida).rolling(window=window).mean()

    rs = media_ganancia / media_perdida
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi
    return df


if __name__ == '__main__':
    df = pd.read_csv(r"../../data/bitget_scrapping.csv")

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

    ### LIMPIEZA DE FORMATO DE DATOS ###

    # Convertir la columna 'Fecha' a datetime
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d")

    # Aplicar la conversión a todas las columnas numéricas
    columnas_monetarias = ["abrir*", "máximo", "mínimo", "cerrar**"]
    for col in columnas_monetarias:
        df[col] = df[col].apply(convertir_euros)

    df["volumen"] = df["volumen"].apply(convertir_volumen)

    # Mostrar resultado
    print("Limpieza de formato de datos:")
    print(df.head())
    print(df.info())

    ### DETECCIÓN Y TRATAMIENTO DE VALORES NULOS ###
    print("\nValores nulos en el dataset:")
    print(df.isnull().sum())

    # Si hay valores nulos en precios o volumen, se imputan con la media
    df.fillna(df.mean(), inplace=True)

    ### DETECCIÓN DE OUTLIERS ###

    print("\nOutliers detectados en el dataset:")
    for col in columnas_monetarias + ["volumen"]:
        outliers = detectar_outliers_iqr(df, col)
        if not outliers.empty:
            print(f"\nOutliers en {col}:")
            print(outliers)

    # No vamos a eliminar outliers ya que pueden ser utiles para la prediccion, solamente los detectamos

    ### RESULTADO FINAL ###
    print("\nDataset limpio y listo para su análisis:")
    print(df.head())



    df["var_diaria"] = df["cerrar**"].pct_change() * 100  # % de variación
    df["volatilidad_50"] = df["cerrar**"].rolling(window=50).std()
    df["volatilidad_200"] = df["cerrar**"].rolling(window=200).std()
    df["rango_dia"] = df["cerrar**"] - df["abrir*"]
    df["momentum_50"] = df["cerrar**"] - df["cerrar**"].shift(50)
    df["momentum_200"] = df["cerrar**"] - df["cerrar**"].shift(200)
    # Media Móvil Simple (SMA)
    df['SMA_50'] = df['cerrar**'].rolling(window=50).mean()  # 50 días
    df['SMA_200'] = df['cerrar**'].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df['EMA_50'] = df['cerrar**'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['cerrar**'].ewm(span=200, adjust=False).mean()

    df = calcular_rsi(df)

    df.to_csv("../../data/limpios_bitget_scrapping.csv", index=False)
