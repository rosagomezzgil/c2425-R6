import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv(r"../../data/halving_data.csv")

    print(df.info())  # Ver tipos de datos y valores nulos
    print(df.describe())  # Estadísticas básicas
    print(df.head())  # Primeras filas

    #convertimos la columna timestamp a datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.date
    df = df.rename(columns={'Fecha': 'timestamp'}) #cambio el nombre para que coincida


    output_path = r"../../data/halving_data_limpio.csv"
    df.to_csv(output_path, encoding = "utf-8-sig", index=False)
    print(df)
