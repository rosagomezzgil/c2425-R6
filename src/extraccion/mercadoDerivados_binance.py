import requests
import pandas as pd
import time
from typing import Optional

def get_binance_futures_historical_data(
        symbol: str = "BTCUSDT",
        days: int = 365,
        interval: str = "1h",
        output_path: Optional[str] = None
):
    """
    Descarga datos históricos de Binance Futures.

    :param symbol: (str) Par de trading (ej. "BTCUSDT").
    :param days: (int) Número de días a recuperar.
    :param interval: (str) Intervalo de las velas ("1m", "5m", "1h", "1d").
    :param output_path: (str, opcional) Ruta para guardar los datos en CSV.

    :return: pd.DataFrame con los datos históricos.
    """

    url = "https://fapi.binance.com/fapi/v1/klines"  # Endpoint de velas de Binance Futures

    all_data = []
    limit = 1000  # Máximo de velas por solicitud en Binance
    interval_ms = {
        "1m": 60 * 1000, "5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000
    }[interval]  # Convertir el intervalo a milisegundos

    end_time = int(time.time() * 1000)  # Tiempo actual en ms
    start_time = end_time - (days * 24 * 60 * 60 * 1000)  # Tiempo inicial en ms

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Lanza error si la solicitud falla
            data = response.json()

            if not data or not isinstance(data, list):
                print("No se obtuvieron más datos o error en la API.")
                break

            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            all_data.append(df[["open", "high", "low", "close", "volume", "trades"]])

            # Avanzamos al último timestamp recibido para evitar duplicados
            start_time = int(df.index[-1].timestamp() * 1000) + interval_ms
            time.sleep(0.5)  # Pausa para evitar bloqueos

        except requests.exceptions.RequestException as e:
            print(f"Error en la solicitud: {e}")
            break

    # Unir los datos descargados
    if all_data:
        full_df = pd.concat(all_data).sort_index()

        if output_path:
            full_df.to_csv(output_path, encoding="utf-8-sig", index=True)
            print(f"Datos guardados en: {output_path}")

        return full_df
    else:
        print("No se obtuvieron datos.")
        return None


if __name__ == '__main__':
    df = get_binance_futures_historical_data(
        symbol="BTCUSDT",
        days=365 * 3,
        interval="1d",
        output_path=r"../../data/extraccion/datos_mercadoDerivados_binance.csv"
    )

    print(df.head())