"""
    IMPORTANTE: para ejecutar este fichero -> EJECUTAR PRIMERO LOS NOTEBOOKS DE ANALISIS

    Este script tiene como objetivo cargar archivos CSV previamente procesados, convertirlos en formato
    Parquet para optimizar su almacenamiento y procesamiento, y finalmente guardar estos archivos en el
    directorio correspondiente.

    1. Carga un archivo CSV (cleaned_data.csv) en un DataFrame de pandas.

    2. Convierte y guarda este DataFrame en formato Parquet con la extensión .parquet para una mejor eficiencia
     en el manejo de grandes volúmenes de datos.

    3. Repite este proceso para otros dos archivos CSV: dataset_7d.csv y dataset_30d.csv.

    4. Al finalizar, imprime las primeras filas de los archivos Parquet resultantes para verificar su contenido.

    Lo guardamos en Parquet por su alta eficiencia en términos de almacenamiento y velocidad de
    lectura/escritura, lo que nos facilita mucho el trabajo para un proyectos de análisis como este con grandes
    conjuntos de datos.
"""

import os
import pandas as pd

if __name__ == "__main__":
    # Cargar el CSV en un DataFrame
    df = pd.read_csv(r"../../data/limpieza/cleaned_data.csv")

    # Guardar como Parquet
    output_path = r"../../data/parquet/data.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    df_parquet = pd.read_parquet(r"../../data/parquet/data.parquet")
    print(df_parquet.head())

    # Cargar el CSV en un DataFrame
    df = pd.read_csv(r"../../data/limpieza/dataset_7d.csv")

    # Guardar como Parquet
    output_path = r"../../data/parquet/data_7d.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    df_parquet = pd.read_parquet(r"../../data/parquet/data_7d.parquet")
    print(df_parquet.head())

    # Cargar el CSV en un DataFrame
    df = pd.read_csv(r"../../data/limpieza/dataset_30d.csv")

    # Guardar como Parquet
    output_path = r"../../data/parquet/data_30d.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    df_parquet = pd.read_parquet(r"../../data/parquet/data_30d.parquet")
    print(df_parquet.head())
