import pandas as pd

if __name__ == "__main__":
    # Cargar el CSV en un DataFrame
    df = pd.read_csv(r"../../data/limpieza/cleaned_data.csv")

    # Guardar como Parquet
    df.to_parquet(r"../../data/parquet/data.parquet", engine="pyarrow")
    df_parquet = pd.read_parquet(r"../../data/parquet/data.parquet")
    print(df_parquet.head())
