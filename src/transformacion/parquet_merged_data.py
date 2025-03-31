# EJECUTAR PRIMERO LOS NOTEBOOKS DE ANALISIS

import pandas as pd

if __name__ == "__main__":
    # Cargar el CSV en un DataFrame
    df = pd.read_csv(r"../../data/limpieza/cleaned_data.csv")

    # Guardar como Parquet
    df.to_parquet(r"../../data/parquet/data.parquet", engine="pyarrow")
    df_parquet = pd.read_parquet(r"../../data/parquet/data.parquet")
    print(df_parquet.head())

    # Cargar el CSV en un DataFrame
    df = pd.read_csv(r"../../data/limpieza/dataset_7d.csv")

    # Guardar como Parquet
    df.to_parquet(r"../../data/parquet/data_7d.parquet", engine="pyarrow")
    df_parquet = pd.read_parquet(r"../../data/parquet/data_7d.parquet")
    print(df_parquet.head())

    # Cargar el CSV en un DataFrame
    df = pd.read_csv(r"../../data/limpieza/dataset_30d.csv")

    # Guardar como Parquet
    df.to_parquet(r"../../data/parquet/data_30d.parquet", engine="pyarrow")
    df_parquet = pd.read_parquet(r"../../data/parquet/data_30d.parquet")
    print(df_parquet.head())
