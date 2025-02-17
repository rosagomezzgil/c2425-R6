import requests
import pandas as pd
from typing import Optional

def get_bitcoin_hashrate(
    timespan: str = "all",
    rolling_average: str = "24hours",
    output_path: Optional[str] = None
):
    """
    Obtiene datos del hashrate de Bitcoin desde Blockchain.com.

    Parámetros:
    - timespan (str): Rango de tiempo (ej. "30days", "1year", "all").
    - rolling_average (str): Promedio móvil (ej. "8hours", "24hours").
    - output_path (str, opcional): Ruta para guardar los datos en CSV.

    Retorna:
    - pd.DataFrame con la información del hashrate.
    """

    url = f"https://api.blockchain.info/charts/hash-rate?timespan={timespan}&rollingAverage={rolling_average}&format=json"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifica errores HTTP

        data = response.json()
        df = pd.DataFrame(data["values"])

        # Convertir timestamp a formato de fecha
        df["x"] = pd.to_datetime(df["x"], unit="s")
        df.rename(columns={"x": "timestamp", "y": "hashrate"}, inplace=True)

        # Guardar CSV si se proporciona una ruta
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Datos guardados en: {output_path}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud: {e}")
        return None


if __name__ == '__main__':
    hashrate_df = get_bitcoin_hashrate(
        output_path="../../data/datos_hashrate.csv"
    )

    print(hashrate_df.head())