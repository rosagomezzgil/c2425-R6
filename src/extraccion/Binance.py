#OHLCV, volumen, precios en vivo de bitcoins
#Binance

import requests
import pandas as pd

def get_binance_data(symbol="BTCUSDT", interval="1d", limit=1000):
    url = "https://api.binance.com/api/v3/klines"

    params = {"symbol": symbol, "interval": interval, "limit": limit}

    response = requests.get(url, params=params)
    assert response.status_code == 200

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
    return df

# Ejecutar la función y ver qué devuelve
binance_df = get_binance_data("BTCUSDT")  # Aseguramos que el símbolo está en mayúsculas

binance_df.to_csv("datos_bitcoin_binance.csv", index=False)
print(binance_df)