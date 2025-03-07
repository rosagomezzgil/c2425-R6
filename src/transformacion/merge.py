import pandas as pd
import os

if __name__ == '__main__':

    """
                                  1 - JUNTAMOS TODOS LOS DATAFRAMES
    """
    ### RUTA ABSOLUTA A LA CARPETA 'DATA' ###
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Sube 2 niveles desde 'src/transformacion/'
    DATA_DIR = os.path.join(BASE_DIR, "data/limpieza")

    ### LISTA DE LOS CSV Q VAMOS A CARGAR ###
    archivos = [
        "limpios_bitget_scrapping.csv", "limpios_binance.csv",
        "limpios_fear_greed.csv", "limpios_coingecko.csv", "limpios_google_trends.csv",
        "limpios_hashrate.csv",
        "limpios_mercadoDerivados_binance.csv", "limpios_mercadoDerivados_deribit.csv", "limpios_sp500.csv",
        "limpios_nasdaq.csv", "limpios_dxy.csv"
    ]

    ### NOMBRES DE SUFIJOS PARA LAS COLUMNAS ###
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

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        print("PERIODO COMPRENDIDO: \n", df["timestamp"].min(), df["timestamp"].max(), "\n")


        #reenombramos columnas para evitar confusiones (excepto 'timestamp')
        df = df.add_suffix(sufijo)  #agrega el sufijo
        df = df.rename(columns={f"timestamp{sufijo}": "timestamp"})  #timestamp sin sufijo

        dataframes.append(df)

    df_final = dataframes[0]
    for df in dataframes[1:]:
        df_final = df_final.merge(df, on="timestamp", how="outer", suffixes=("", ""))

    #Guardar el DataFrame final
    output_path = os.path.join(DATA_DIR, "merged_data.csv")
    df_final.to_csv(output_path, index=False)