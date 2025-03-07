from Calcular_RSI import calcular_rsi
import pandas as pd

if __name__ == '__main__':
    # Cargar el DataFrame
    df = pd.read_csv(r"../../data/extraccion/datos_mercadoDerivados_deribit.csv", parse_dates=["timestamp"])

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

    ###CAMBIO DE NOMBRE DE LAS COLUMNAS PARA EL MERGE POSTERIORMENTE
    df_diario = df_diario.rename(columns={'open': 'mDopen',
                           'high': 'mDhigh',
                           'low': 'mDlow',
                           'close': 'mDclose',
                           'volume': 'mDvolume'})


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
    df_diario['mDSMA_50'] = df_diario['mDclose'].rolling(window=50).mean()  # 50 días
    df_diario['mDSMA_200'] = df_diario['mDclose'].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df_diario['mDEMA_50'] = df_diario['mDclose'].ewm(span=50, adjust=False).mean()
    df_diario['mDEMA_200'] = df_diario['mDclose'].ewm(span=200, adjust=False).mean()

    df_diario = calcular_rsi(df_diario, 'mDclose')
    df_diario = df_diario.rename(columns={'RSI': 'mD_RSI'})

    # Guardar el conjunto de datos
    output_path = r"../../data/limpieza/limpios_mercadoDerivados_deribit.csv"
    df_diario.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df_diario)
