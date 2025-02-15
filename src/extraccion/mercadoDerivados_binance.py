import requests
import pandas as pd
import time

# Función para obtener datos históricos de Binance Futures
def get_binance_futures_historical_data(symbol="BTCUSDT", days=365, interval="1h"):
    url = "https://fapi.binance.com/fapi/v1/klines"  # Endpoint de velas de Binance Futures

    all_data = []
    limit = 1000  # Máximo de velas por solicitud en Binance
    interval_ms = 60 * 60 * 1000  # 1 hora en milisegundos
    end_time = int(time.time() * 1000)  # Tiempo actual en milisegundos
    start_time = end_time - (days * 24 * 60 * 60 * 1000)  # Hace un año en milisegundos

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit  # Máximo permitido por la API
        }

        response = requests.get(url, params=params)
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Guardamos solo las columnas necesarias
            all_data.append(df[["open", "high", "low", "close", "volume"]])

            # Avanzamos al último tiempo recibido para evitar duplicados
            start_time = int(df.index[-1].timestamp() * 1000) + interval_ms
        else:
            print("No se obtuvieron más datos o error en la API.")
            break

        time.sleep(0.5)  # Pequeña pausa para evitar bloqueos

    # Unir todos los datos y ordenarlos
    if all_data:
        full_df = pd.concat(all_data)
        return full_df.sort_index()
    else:
        print("No se obtuvieron datos.")
        return None

# Llamada a la función para obtener 1 año de datos en intervalos de 1 hora
df = get_binance_futures_historical_data(days=365, interval="1h")
print(df.head())

# Guardar el conjunto de datos
output_path = r"../../data/datos_mercadoDerivados_binance.csv"
df.to_csv(output_path, encoding="utf-8-sig", index=True)

