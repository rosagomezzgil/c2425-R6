import requests
import pandas as pd
from typing import Optional

def get_fear_greed_index(limit: int = 0, date_format: str = "world", output_path: Optional[str] = None):
    """
    Obtiene el índice de Miedo y Codicia desde la API de alternative.me.

    Parámetros:
    - limit (int): Número de registros a obtener (0 para todos).
    - date_format (str): Formato de fecha ('world' o 'us').
    - output_path (str, opcional): Ruta donde guardar el CSV.

    Retorna:
    - pd.DataFrame con los datos obtenidos.
    """

    url = f"https://api.alternative.me/fng/?limit={limit}&format=json&date_format={date_format}"

    response = requests.get(url)
    response.raise_for_status()  # Verifica si la solicitud fue exitosa
    data = response.json().get("data", [])

    df = pd.DataFrame(data)

    # Convertir 'timestamp' a formato de fecha, asegurando que se interprete correctamente
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=(date_format == "world"), errors='coerce')

    # Seleccionar y renombrar columnas
    df = df[["timestamp", "value"]]
    df.rename(columns={"value": "fear_greed_index"}, inplace=True)

    # Asegurar que el índice de miedo y codicia sea numérico
    df["fear_greed_index"] = pd.to_numeric(df["fear_greed_index"], errors='coerce')

    # Guardar CSV si se proporciona una ruta
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Datos guardados en: {output_path}")

    return df

if __name__ == '__main__':
    fear_greed_df = get_fear_greed_index(output_path="../../data/extraccion/datos_bitcoin_fear_greed.csv")
    print(fear_greed_df.shape)
    print(fear_greed_df.head())