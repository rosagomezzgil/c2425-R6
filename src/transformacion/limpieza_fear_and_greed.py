import pandas as pd

if __name__ == '__main__':
    # Cargar datos desde un CSV (ajusta la ruta a tu archivo)
    df = pd.read_csv(r"../../data/datos_bitcoin_fear_greed.csv")

    # Ver las primeras filas
    print(df.head())
    #Comprobar el tipo de las variables
    print(df.dtypes)

    #Convertir timestamp a fecha
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convertir a formato fecha
    df.sort_values('timestamp', inplace=True)  # Ordenar por fecha

    # Ver las primeras filas
    print(df.head())
    # Comprobar el tipo de las variables
    print(df.dtypes)


    print(df.isnull().sum()) #no hay nulos

    df.to_csv("../../data/datos_bitcoin_limpiados_fear_greed.csv", index=False)