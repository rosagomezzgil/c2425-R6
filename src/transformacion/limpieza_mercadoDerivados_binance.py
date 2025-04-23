"""
    Este script procesa el dataset de mercado de derivados de Binance.
    Para ello se carga del archivo CSV y se hace una eliminación de valores duplicados y valores nulos,
    así como un renombrado de variables clave para diferenciar métricas específicas del mercado de
    derivados (prefijo 'mD').

    Además, se calculan nuevas métricas como variación diaria, volatilidad, momentum, medias móviles
    (SMA y EMA) y el RSI utilizando la función ya definida, preparando los datos para su uso en modelos
    de predicción.

    Por úlimo, se extrae el dataset limpio a un nuevo archivo CSV, listo para su integración con datos
    del mercado spot.

    """



from Calcular_RSI import calcular_rsi
from utils.paths import get_path
import pandas as pd
import os

if __name__ == "__main__":
    df = pd.read_csv(get_path('data', 'extraccion', 'datos_mercadoDerivados_binance.csv'))

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

    # Mostrar resultado
    print("Limpieza de formato de datos:")
    print(df.head())
    print(df.info())

    ### DETECCIÓN Y TRATAMIENTO DE VALORES NULOS ###
    print("\nValores nulos en el dataset:")
    print(df.isnull().sum())

    ### CAMBIO DE NOMBRES DE LAS VARIABLES PARA EL ESTUDIO ###
    df = df.rename(
        columns={
            "open": "mDopen",
            "high": "mDhigh",
            "low": "mDlow",
            "close": "mDclose",
            "volume": "mDvolume",
            "trades": "mDtrades",
        }
    )

    ### DETECCIÓN DE OUTLIERS ###
    # No vamos a eliminar outliers ya que pueden ser utiles para la prediccion, solamente los detectamos

    ### RESULTADO FINAL ###
    print("\nDataset limpio y listo para su análisis:")
    print(df.head())

    df["mDvar_diaria"] = df["mDclose"].pct_change() * 100
    df["mDvolatilidad_50"] = df["mDclose"].rolling(window=50).std()
    df["mDvolatilidad_200"] = df["mDclose"].rolling(window=200).std()
    df["mDrango_dia"] = df["mDclose"] - df["mDlow"]
    df["mDmomentum_50"] = df["mDclose"] - df["mDclose"].shift(50)
    df["mDmomentum_200"] = df["mDclose"] - df["mDclose"].shift(200)
    # Media Móvil Simple (SMA)
    df["mDSMA_50"] = df["mDclose"].rolling(window=50).mean()  # 50 días
    df["mDSMA_200"] = df["mDclose"].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df["mDEMA_50"] = df["mDclose"].ewm(span=50, adjust=False).mean()
    df["mDEMA_200"] = df["mDclose"].ewm(span=200, adjust=False).mean()

    df = calcular_rsi(df, "mDclose")
    df = df.rename(columns={"RSI": "mD_RSI"})

    output_path = get_path('data', 'limpieza', 'limpios_mercadoDerivados_binance.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(
        output_path,
        encoding="utf-8-sig",
        index=False,
    )
    print(df)
