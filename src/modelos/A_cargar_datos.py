import pandas as pd

def cargar_datos_7d(parquet_file: str):
    df = pd.read_csv(parquet_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def cargar_datos_30d(parquet_path: str) -> pd.DataFrame:
    df = pd.read_csv(parquet_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('D')  # frecuencia diaria
    return df