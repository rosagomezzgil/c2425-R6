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
    #faltan sp500, nasdaq, sentiment analysis. Añadir y añadir lo correspondiente a ellos en el resto de codigo

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
    # REVISAR CANTIDAD DE AÑOS (muchos nulos porq hay fuentes que tienen muy pocos datos)
    df_final = df_final[df_final["timestamp"] >= "2020-01-01"]

    ### COLUMNAS QUE VAMOS A CONSOLIDAR (juntar) ###
    columns_to_merge = ["open", "close", "RSI", "SMA_50", "SMA_200",
                        "EMA_50", "EMA_200", "volatilidad_50",
                        "volatilidad_200", "momentum_50", "momentum_200", #mercado Spot
                        "mDopen", "mDclose", "mD_RSI", "mDSMA_50", "mDSMA_200",
                        "mDEMA_50", "mDEMA_200", "mDvolatilidad_50",
                        "mDvolatilidad_200", "mDmomentum_50", "mDmomentum_200"] #mercado de Derivados
    #faltan VOLUMENES (de mercado spot y mercados de derivados, porque no se juntan por la media, más abajo añadidos)
    #NO se pueden juntar datos de mercado spot (solo valores de bitcoin) con los del mercado de derivados

    #creamos un nuevo DataFrame con solo timestamp (iremos metiendo aqui los datos del df LIMPIO)
    ###  df_cleaned ES EL DEFINITIVO (df_final es donde tenemos todos los datos pero sin organizar)
    df_cleaned = df_final[["timestamp"]].copy()


    """Juntamos las columnas que son lo mimso mediante la media entre sus valores"""
    #consolidamos cada grupo de columnas
    for base_name in columns_to_merge:
        matching_cols = [col for col in df_final.columns if col.startswith(base_name)]

        if matching_cols:
            #usar la media entre columnas si hay más de una fuente, ignorando NaN
            df_cleaned[base_name] = df_final[matching_cols].mean(axis=1, skipna=True)



    #agregamos columnas útiles que no tuvimos en cuenta a la hora de consolidar
    keep_columns = ["fear_greed_index_fg", "market_cap_coingecko", "bitcoin_gtrends",
                    "hashrate_hashrate"]


    df_cleaned = pd.concat([df_cleaned, df_final[keep_columns]], axis=1)

    #FALTAN MUCHAS COLUMNAS, COMO LOS VOLÚMENES, MAX Y MIN, VARIACIONES DIARIAS, ETC
    #ADICION DE ESAS COLUMNAS:

    """VOLUMENES"""
    ### VOLUMEN DE BTC MOVIDO ESE DÍA ###
    #mercado spot
    df_cleaned["volume"] = df_final["volume_coingecko"] #ESTA EN USD($)
    #COINGECKO CONTIENE EL VOLUMEN TOTAL QUE SE HA MOVIDO EN EL MERCADO ENTERO (todas las paginas de exchange)
    #MIENTRAS QUE BINANCE Y BITGET SÓLO LA CANTIDAD QUE SE HA MOVIDO EN SU PLATAFORMA
    #SUMAR ESTOS DATOS PODRÍA PROVOCAR DUPLICACION DE CANTIDADES

    #mercado de derivados
    #(primero voy a pasar ambos mercados de derivados a USD($) para que los datos sean consistentes
    df_final["mDvolume_mDbinance"] = df_final['mDvolume_mDbinance'] * df_final['mDclose_mDbinance']
    df_final = df_final.rename(columns={'mDvolume_mDbinance': 'mDvolume_mDbinance_usd'})

    df_final["mDvolume_mDderibit"] = df_final['mDvolume_mDderibit'] * df_final['mDclose_mDderibit']
    df_final = df_final.rename(columns={'mDvolume_mDderibit': 'mDvolume_mDderibit_usd'})

    df_cleaned["mDvolume"] = df_final['mDvolume_mDbinance_usd'] + df_final['mDvolume_mDderibit_usd'] #en USD($)


    """MAXIMOS, MINIMOS"""
    #mercado spot
    df_cleaned['high'] = df_final[['high_bitget', 'high_binance_', 'close_coingecko']].max(axis=1)
    df_cleaned['low'] = df_final[['low_bitget', 'low_binance_', 'close_coingecko']].min(axis=1)
    #pongo close_coingecko ya que es el unico dato de precio de esa fuente de datos

    #mercado de derivados
    df_cleaned['mDhigh'] = df_final[['mDhigh_mDbinance', 'mDhigh_mDderibit']].max(axis=1)
    df_cleaned['mDlow'] = df_final[['mDlow_mDbinance', 'mDlow_mDderibit']].min(axis=1)



    """VARIACIÓN DIARIA"""
    #mercado spot
    df_cleaned["var_diaria"] = df_cleaned["close"].pct_change() * 100  # % de variación

    #mercado de Derivados
    df_cleaned["mDvar_diaria"] = df_cleaned["mDclose"].pct_change() * 100



    """RANGO DIARIO"""
    #mercado spot
    df_cleaned["rango_dia"] = df_cleaned["close"].diff().abs()

    #mercado de Derivados
    df_cleaned["mDrango_dia"] = df_cleaned["mDclose"] - df_cleaned["mDopen"]




    ### REORDENAMOS LAS COLUMNAS DEL DATAFRAME (primero mercado spot, luego m.Derivados, luego columnas sueltas)
    print(df_cleaned.columns.tolist())
    nuevo_orden = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'var_diaria', 'rango_dia', 'RSI',
                   'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'volatilidad_50', 'volatilidad_200', 'momentum_50',
                   'momentum_200',
                   #mercado SPOt
                   'mDopen', 'mDhigh', 'mDlow', 'mDclose', 'mDvolume', 'mDvar_diaria', 'mDrango_dia', 'mD_RSI',
                   'mDSMA_50', 'mDSMA_200', 'mDEMA_50', 'mDEMA_200', 'mDvolatilidad_50', 'mDvolatilidad_200',
                   'mDmomentum_50', 'mDmomentum_200',
                   #mercado DERIVADOS
                   'fear_greed_index_fg', 'market_cap_coingecko', 'bitcoin_gtrends', 'hashrate_hashrate',
                   #COLUMNAS SUELTAS
                    ]

    df_cleaned = df_cleaned[nuevo_orden]



    df_cleaned = df_cleaned.reset_index(drop=True)
    print(df_cleaned.dtypes)
    print(df_cleaned.info())
    print(df_cleaned.isnull().sum())
    print(df_cleaned.head())



    #Guardar el DataFrame final
    output_path = os.path.join(DATA_DIR, "merged_data.csv")
    #df_final.to_csv(output_path, index=False)
    df_cleaned.to_csv(output_path, index=False)