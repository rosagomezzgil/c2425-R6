import pandas as pd

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

    df.to_csv("../../data/bitget_scrapping.csv", index=False)
