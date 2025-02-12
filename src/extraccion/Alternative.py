import requests
import pandas as pd

def get_fear_greed_index():
    url = "https://api.alternative.me/fng/?limit=0&format=json&date_format=world"

    response = requests.get(url)
    response.raise_for_status()  # Verifica si la solicitud fue exitosa
    data = response.json().get("data", [])

    df = pd.DataFrame(data)

    # Convertir 'timestamp' a formato de fecha, asegurando que se interprete correctamente
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors='coerce')

    # Seleccionar y renombrar columnas
    df = df[["timestamp", "value"]]
    df.rename(columns={"value": "fear_greed_index"}, inplace=True)

    # Asegurar que el índice de miedo y codicia sea numérico
    df["fear_greed_index"] = pd.to_numeric(df["fear_greed_index"], errors='coerce')

    return df


fear_greed_df = get_fear_greed_index()
fear_greed_df.to_csv("datos_bitcoin_fear_greed.csv", index=False)
print(fear_greed_df)