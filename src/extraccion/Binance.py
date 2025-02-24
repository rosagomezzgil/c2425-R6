#OHLCV, volumen, precios en vivo de bitcoins
#Binance

import requests
import pandas as pd
from typing import Optional

def get_binance_data(
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        output_path: Optional[str] = None
):
    """
    Obtiene datos históricos de Binance.

    :param symbol: (str) Par de trading (ej. 'BTCUSDT').
    :param interval: (str) Intervalo de tiempo ('1m', '1h', '1d', etc.).
    :param limit: (int) Número de registros a obtener.
    :param start_time: (int, opcional) Tiempo de inicio en milisegundos.
    :param end_time: (int, opcional) Tiempo de fin en milisegundos.
    :param output_path: (str, opcional) Ruta donde guardar el CSV.
    :return: pd.DataFrame con los datos obtenidos.
    """

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)
    response.raise_for_status()  # Lanza un error si la solicitud falla
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    #Cambiar los tipos de datos
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    #Seleccionar solo las columnas necesarias
    df = df[["timestamp", "open", "high", "low", "close", "volume", "trades"]]
    df.columns = ["timestamp", "open_binance", "high_binance", "low_binance", "close_binance", "volume_binance", "trades_binance"]

    # Guardar CSV si se proporciona una ruta
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Datos guardados en: {output_path}")

    return df

if __name__ == '__main__':
    binance_df = get_binance_data(
        symbol="BTCUSDT",
        output_path="../../data/datos_bitcoin_binance.csv"
    )

    print(binance_df.head())