import pandas as pd

if __name__ == '__main__':
    # Cargar datos desde un CSV (ajusta la ruta a tu archivo)
    df = pd.read_csv(r"../../data/datos_hashrate.csv")

    # Ver las primeras filas
    print(df.head())
    #Comprobar el tipo de las variables
    print(df.dtypes)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values('timestamp', inplace=True)  # Ordenar por fecha

    # Ver si hay valores nulos
    print(df.isnull().sum())  # No tiene valores nulos

    df.to_csv("../../data/datos_hashrate_limpiados_binance.csv", index=False)