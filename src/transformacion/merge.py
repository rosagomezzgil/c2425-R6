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
        "datos_hashrate_limpiados_binance.csv",
        "mercadoDerivados_binance_limpio.csv", "limpieza_mercDer_deribit.csv", "datos_bitcoin_limpiados_sp500.csv",
        "datos_bitcoin_limpiados_nasdaq.csv", "datos_bitcoin_limpiados_dxy.csv"
    ]

    ### NOMBRES DE SUFIJOS PARA LAS COLUMNAS (a quitar después, en la limpieza) ###
    sufijos = [
        "_bitget", "_binance", "_fg", "_coingecko", "_gtrends",
        "_hashrate", "_mDbinance", "_mDderibit", "_sp500", "_nasdaq", "_dxy"
    ]

    ### ENCONTRAR LOS ARCHIVOS EN 'data/' QUE COINCIDAN CON LOS NOMBRES SELECCIONADOS ###
    csv_files = [os.path.join(DATA_DIR, file) for file in archivos if
                 os.path.exists(os.path.join(DATA_DIR, file))]

    ### CARGAMOS LOS ARCHIVOS Y MOSTRAMOS LAS ESTRUCTURAS QUE TIENEN ###
    dataframes = []
    for file, sufijo in zip(csv_files, sufijos):
        df = pd.read_csv(file)

        # mostramos información de las columnas
        print(f"Archivo: {os.path.basename(file)}")

        assert ("timestamp" in df.columns)
        print("PERIODO COMPRENDIDO: \n", df["timestamp"].min(), df["timestamp"].max(), "\n")


        df["timestamp"] = pd.to_datetime(df["timestamp"])

        #reenombramos columnas para evitar confusiones (excepto 'timestamp')
        df = df.add_suffix(sufijo)  #agrega el sufijo
        df = df.rename(columns={f"timestamp{sufijo}": "timestamp"})  #timestamp sin sufijo

        dataframes.append(df)

    df_final = dataframes[0]
    for df in dataframes[1:]:
        df_final = df_final.merge(df, on="timestamp", how="outer", suffixes=("", ""))

    #Guardar el DataFrame final
    output_path = os.path.join(DATA_DIR, "merged_data.csv")
    #df_final.to_csv(output_path, index=False)
    df_final.to_csv(output_path, index=False)