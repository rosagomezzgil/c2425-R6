"""
    Este script se encarga de limpiar y transformar los datos de Bitcoin obtenidos a través
    de scraping desde la plataforma Bitget. Primero convierte los formatos de fecha y
    valores monetarios (incluyendo volúmenes expresados en billones) a tipos numéricos.

    También se renombran columnas para mantener consistencia con otros datasets.
    Se eliminan duplicados y valores nulos, y se detectan outliers usando el método IQR,
    aunque no se eliminan porque pueden ser muy relevantes en análisis predictivos.

    Además, se calculan nuevas métricas como variación diaria, volatilidad,
    momentum, medias móviles (SMA y EMA) y el RSI utilizando la función
    ya definida, preparando los datos para su uso en modelos de predicción.

    Finalmente, el dataset limpio se guarda en un nuevo archivo CSV.

    """


from Calcular_RSI import calcular_rsi
import pandas as pd
import os

def convertir_euros(valor):
    """
    Función para convertir valores monetarios en float
    """
    if isinstance(valor, str):
        valor = valor.replace("€", "").replace("$", "").replace(",", "")
        return float(valor)
    return valor


def convertir_volumen(vol):
    """
    Función para convertir volumen (B de Billones a número real)
    """
    if (
        isinstance(vol, str) and "B" in vol
    ):  # Solo aplicar reemplazos si vol es un string
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


if __name__ == "__main__":
    df = pd.read_csv(r"../data/extraccion/bitget_scrapping.csv")

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
    df.sort_values("fecha", inplace=True)  # Ordenar por fecha

    # cambio el nombre de la columna fecha a timpestamp para que tenga formato y nombre igual que el resto
    df = df.rename(columns={"fecha": "timestamp"})
    df = df.rename(columns={"abrir*": "open"})
    df = df.rename(columns={"máximo": "high"})
    df = df.rename(columns={"mínimo": "low"})
    df = df.rename(columns={"cerrar**": "close"})

    # Aplicar la conversión a todas las columnas numéricas
    columnas_monetarias = ["open", "high", "low", "close"]
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

    df["var_diaria"] = df["close"].pct_change() * 100  # % de variación
    df["volatilidad_50"] = df["close"].rolling(window=50).std()
    df["volatilidad_200"] = df["close"].rolling(window=200).std()
    df["rango_dia"] = df["high"] - df["low"]
    df["momentum_50"] = df["close"] - df["close"].shift(50)
    df["momentum_200"] = df["close"] - df["close"].shift(200)
    # Media Móvil Simple (SMA)
    df["SMA_50"] = df["close"].rolling(window=50).mean()  # 50 días
    df["SMA_200"] = df["close"].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["close"].ewm(span=200, adjust=False).mean()

    df = calcular_rsi(df, "close")

    output_path = r"../data/limpieza/limpios_bitget_scrapping.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
