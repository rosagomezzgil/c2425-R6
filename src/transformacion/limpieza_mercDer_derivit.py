"""
    Este script procesa el dataset de mercado de derivados de Deribit.
    Para ello se carga del archivo CSV y se hace una conversión del timestamp a formato fecha (si
    es necesario) y re-muestreo de los datos para obtener los valores diarios y un renombrado de
    las columnas para diferenciar métricas específicas del mercado de derivados (prefijo 'mD').

    Además, se calculan nuevas métricas como variación diaria, volatilidad, momentum, medias móviles
    (SMA y EMA) y el RSI utilizando la función ya definida, preparando los datos para su uso en modelos
    de predicción.

    Por úlimo, se extrae el dataset limpio a un nuevo archivo CSV, listo para su integración con datos
    del mercado spot.

    """

from Calcular_RSI import calcular_rsi
import pandas as pd
import os

if __name__ == "__main__":
    # Cargar el DataFrame
    df = pd.read_csv(
        r"../../data/extraccion/datos_mercadoDerivados_deribit.csv",
        parse_dates=["timestamp"],
    )

    # Convertir timestamp a datetime si aún no lo es
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Agrupar por día y calcular los valores relevantes
    df_diario = (
        df.resample("D", on="timestamp")
        .agg(
            {
                "open": "first",  # Primer precio del día
                "high": "max",  # Precio máximo del día
                "low": "min",  # Precio mínimo del día
                "close": "last",  # Último precio del día
                "volume": "sum",  # Volumen total del día
            }
        )
        .reset_index()
    )

    ###CAMBIO DE NOMBRE DE LAS COLUMNAS PARA EL MERGE POSTERIORMENTE
    df_diario = df_diario.rename(
        columns={
            "open": "mDopen",
            "high": "mDhigh",
            "low": "mDlow",
            "close": "mDclose",
            "volume": "mDvolume",
        }
    )

    df_diario = df_diario.sort_values(by="timestamp")

    # Verificar resultado
    print(df_diario.head())
    ### DETECCIÓN Y TRATAMIENTO DE VALORES NULOS ###
    print("\nValores nulos en el dataset:")
    print(df.isnull().sum())

    df_diario["mDvar_diaria"] = df_diario["mDclose"].pct_change() * 100
    df_diario["mDvolatilidad_50"] = df_diario["mDclose"].rolling(window=50).std()
    df_diario["mDvolatilidad_200"] = df_diario["mDclose"].rolling(window=200).std()
    df_diario["mDrango_dia"] = df_diario["mDclose"] - df_diario["mDopen"]
    df_diario["mDmomentum_50"] = df_diario["mDclose"] - df_diario["mDclose"].shift(50)
    df_diario["mDmomentum_200"] = df_diario["mDclose"] - df_diario["mDclose"].shift(200)
    # Media Móvil Simple (SMA)
    df_diario["mDSMA_50"] = df_diario["mDclose"].rolling(window=50).mean()  # 50 días
    df_diario["mDSMA_200"] = df_diario["mDclose"].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df_diario["mDEMA_50"] = df_diario["mDclose"].ewm(span=50, adjust=False).mean()
    df_diario["mDEMA_200"] = df_diario["mDclose"].ewm(span=200, adjust=False).mean()

    df_diario = calcular_rsi(df_diario, "mDclose")
    df_diario = df_diario.rename(columns={"RSI": "mD_RSI"})

    # Guardar el conjunto de datos
    output_path = r"../../data/limpieza/limpios_mercadoDerivados_deribit.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_diario.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df_diario)
