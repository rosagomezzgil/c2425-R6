import pandas as pd
import glob
import os

if __name__ == '__main__':

    """
                                  1 - JUNTAMOS TODOS LOS DATAFRAMES
    """

    ### RUTA ABSOLUTA A LA CARPETA 'DATA' ###
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Sube 2 niveles desde 'src/transformacion/'
    DATA_DIR = os.path.join(BASE_DIR, "data")

    ### LISTA DE LOS CSV Q VAMOS A CARGAR ###
    archivos = [
        "limpios_bitget_scrapping.csv", "creacion_variables_binance.csv",
        "datos_bitcoin_limpiados_fear_greed.csv", "datos_coingecko_limpios.csv", "limpios_google_trends.csv",
        "halving_data_limpio.csv", "datos_hashrate_limpiados_binance.csv",
        "mercadoDerivados_binance_limpio.csv", "limpieza_mercDer_deribit.csv"
    ]

    ### NOMBRES DE SUFIJOS PARA LAS COLUMNAS (a quitar después, en la limpieza) ###
    sufijos = [
        "_bitget", "_", "_fg", "_coingecko", "_gtrends",
        "_halving", "_hashrate", "_mDbinance", "_mDderibit"
    ]

    ### ENCONTRAR LOS ARCHIVOS EN 'data/' QUE COINCIDAN CON LOS NOMBRES SELECCIONADOS ###
    csv_files = [os.path.join(DATA_DIR, file) for file in archivos if
                 os.path.exists(os.path.join(DATA_DIR, file))]

    ### CARGAMOS LOS ARCHIVOS Y MOSTRAMOS LAS ESTRUCTURAS QUE TIENEN ###
    dataframes = []
    for file, sufijo in zip(csv_files, sufijos):
        df = pd.read_csv(file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        #reenombramos columnas para evitar confusiones (excepto 'timestamp')
        df = df.add_suffix(sufijo)  #agrega el sufijo
        df = df.rename(columns={f"timestamp{sufijo}": "timestamp"})  #timestamp sin sufijo

        dataframes.append(df)

        #mostramos información de las columnas
        print(f"\n📂 Archivo: {os.path.basename(file)}")
        print(df.dtypes)  #mostramos los tipos de las columnas
        print(df.head())
        print("PERIODO COMPRENDIDO: \n", df["timestamp"].min(), df["timestamp"].max(), "\n")

    df_final = dataframes[0]
    for df in dataframes[1:]:
        df_final = df_final.merge(df, on="timestamp", how="outer", suffixes=("", ""))



    """
    Tal cual así, el dataframe está sin depurar, con absolutamente todos los datos de todos los dataframes
    de datos. Esto no es lo que queremos, pues los datos están sin organizar y metidos a lo bruto en el df,
    así que vamos a simplificar, limpiar el dataframe y depurarlo para que sea válido para realizar el estudio 
    sobre él. 
    """

    #resumen general del dataframe
    print(df_final.dtypes)
    print(df_final.info())
    print(df_final.head())
    print(df_final.columns.tolist())

    #contar valores nulos
    missing_values = df_final.isnull().sum()
    print(missing_values)

    #verificamos fechas únicas y rango temporal
    date_range = (df_final["timestamp"].min(), df_final["timestamp"].max())
    unique_dates = df_final["timestamp"].nunique()
    print(date_range)
    print(unique_dates)


    """
    Vamos a eliminar fechas, ya que hay muchas que son irrelevantes para el estudio. Vamos a fusionar columnas que 
    tengan relación directa (o que sean lo mismo pero de distintas fuentes de datos), eliminar columnas con
    demasiados nulos.
    """

    """
                                  2 - LIMPIEZA DEL DATAFRAME CONJUNTO
    """

    ### ELIMINAMOS FECHAS IRRELEVANTES (mantenemos datos desde 2020 en adelante, últimos 5 años) ###
    # REVISAR CANTIDAD DE AÑOS
    df_final = df_final[df_final["timestamp"] >= "2020-01-01"]

    ### COLUMNAS QUE VAMOS A CONSOLIDAR (juntar) ###
    columns_to_merge = ["close", "open", "RSI", "SMA_50", "SMA_200",
                        "EMA_50", "EMA_200", "volatilidad_50",
                        "volatilidad_200", "momentum_50", "momentum_200", #mercado Spot
                        "mDclose", "mDopen", "mD_RSI", "mDSMA_50", "mDSMA_200",
                        "mDEMA_50", "mDEMA_200", "mDvolatilidad_50",
                        "mDvolatilidad_200", "mDmomentum_50", "mDmomentum_200"] #mercado de Derivados
    #faltan VOLUMENES (de mercado spot y mercados de derivados)
    #no se pueden juntar datos de mercado spot (solo valores de bitcoin) con los del mercado de derivados

    #creamos un nuevo DataFrame con solo timestamp
    df_cleaned = df_final[["timestamp"]].copy()


    """Juntamos las columnas que son lo mimso mediante la media entre sus valores"""
    #consolidamos cada grupo de columnas
    for base_name in columns_to_merge:
        matching_cols = [col for col in df_final.columns if col.startswith(base_name)]

        if matching_cols:
            #usar la media entre columnas si hay más de una fuente, ignorando NaN
            df_cleaned[base_name] = df_final[matching_cols].mean(axis=1, skipna=True)



    #agregamos columnas útiles que no tuvimos en cuenta a la hora de consolidar
    keep_columns = ["fear_greed_index_fg", "price_coingecko", "market_cap_coingecko", "bitcoin_gtrends",
                    "hashrate_hashrate"]


    df_cleaned = pd.concat([df_cleaned, df_final[keep_columns]], axis=1)

    #FALTAN MUCHAS COLUMNAS, COMO LOS MAXIMOS, MINIMOS, VOLUMENES, VARIACIONES DIARIAS, ETC
    #REVISAR


    df_cleaned = df_cleaned.reset_index(drop=True)
    print(df_cleaned.dtypes)
    print(df_cleaned.info())
    print(df_cleaned.isnull().sum())
    print(df_cleaned.head())



    #Guardar el DataFrame final
    output_path = os.path.join(DATA_DIR, "merged_data.csv")
    #df_final.to_csv(output_path, index=False)
    df_cleaned.to_csv(output_path, index=False)