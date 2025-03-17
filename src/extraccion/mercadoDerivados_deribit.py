import requests
import pandas as pd
import time
from typing import Optional


def get_deribit_historical_data(
    instrument: str = "BTC-PERPETUAL",
    days: int = 365,
    interval: str = "1h",
    resolution: str = "60",
    output_path: Optional[str] = None,
):
    """
    Descarga datos históricos de Deribit

    :param instrument: (str)
    :param days: (int) Número de días a recuperar
    :param interval: (str) Intervalo de las velas ("1m", "5m", "1h", "1d").
    :param resolution: (str) Resolución: "1", "5", "15", "60", etc.
    :param output_path: (str, opcional) Ruta para guardar los datos en CSV.

    :return: pd.DataFrame cons los datos históricos
    """

    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"

    all_data = []
    interval_ms = {
        "1m": 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }[
        interval
    ]  # Convertir el intervalo a milisegundos

    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    current_start = start_time

    while current_start < end_time:
        current_end = min(current_start + interval_ms, end_time)
        print(f"Cuanto falta: {end_time - current_end}")

        params = {
            "instrument_name": instrument,
            "start_timestamp": current_start,
            "end_timestamp": current_end,
            "resolution": resolution,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Lanza error si la solicitud falla
            data = response.json()

            if "result" not in data or not data["result"]:
                print("Error al obtener datos en el rango:", data)

            df = pd.DataFrame(data["result"])

            df["timestamp"] = pd.to_datetime(
                df["ticks"], unit="ms"
            )  # Convertir timestamp a fecha legible
            df.set_index("timestamp", inplace=True)

            all_data.append(df[["open", "high", "low", "close", "volume"]])

            # Avanzar al siguiente intervalo
            current_start += interval_ms
            time.sleep(0.5)  # Evitar ser bloqueado por la API

        except requests.exceptions.RequestException as e:
            print(f"Error en la solicitud: {e}")
            break

    # Concatenar todos los datos obtenidos
    if all_data:
        full_df = pd.concat(all_data).sort_index()

        if output_path:
            full_df.to_csv(output_path, encoding="utf-8-sig", index=True)
            print(f"Datos guardados en: {output_path}")

        return full_df
    else:
        print("No se obtuvieron datos.")
        return None


if __name__ == "__main__":
    df = get_deribit_historical_data(
        days=365 * 3,
        interval="1d",
        output_path=r"../../data/extraccion/datos_mercadoDerivados_deribit.csv",
    )

    print(df.shape)
    print(df.head())
