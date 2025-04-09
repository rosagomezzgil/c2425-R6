"""
NO EJECUTAR NO FORMA PARTE DEL PROYECTO (por ahora)

Descripción:
Este script se conecta a la API de Binance para obtener datos del mercado de criptomonedas en tiempo real. En este caso, extrae información sobre el par de trading "BTCUSDT" (Bitcoin a USDT), pero el símbolo puede ser modificado. Los datos obtenidos incluyen el precio actual, el cambio de precio, el volumen de transacciones y otros indicadores relevantes. La información se guarda en un archivo CSV, agregando nuevas entradas cada minuto sin sobrescribir los datos anteriores.

Funcionalidad:
1. Obtención de datos: Utiliza la API pública de Binance para obtener información sobre el mercado de criptomonedas.
2. Procesamiento de datos: Los datos obtenidos son convertidos en un `DataFrame` de pandas, con un timestamp y varios campos relacionados con el mercado.
3. Almacenamiento en CSV: Los datos extraídos se guardan en un archivo CSV en la ruta especificada, añadiendo nuevas filas sin borrar las anteriores.
4. Recolección periódica: El script se ejecuta en un bucle infinito, recolectando y almacenando datos cada 60 segundos.
"""

import requests
import pandas as pd
import time


# Función para obtener Market Data
def get_market_data(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        market_data = {
            "timestamp": pd.Timestamp.now(),
            "symbol": symbol,
            "price": float(data["lastPrice"]),
            "priceChange": float(data["priceChange"]),
            "priceChangePercent": float(data["priceChangePercent"]),
            "highPrice": float(data["highPrice"]),
            "lowPrice": float(data["lowPrice"]),
            "volume": float(data["volume"]),
            "quoteVolume": float(data["quoteVolume"]),
        }
        return pd.DataFrame([market_data])
    else:
        print("Error al obtener datos:", response.status_code)
        return None


# Archivo donde se guardarán los datos
csv_filename = r"../../data/extraccion/datos_hashrate.csv"

# Bucle para recolectar datos cada X segundos
while True:
    df = get_market_data("BTCUSDT")

    if df is not None:
        # Guardar los datos en CSV sin borrar los anteriores
        df.to_csv(
            csv_filename,
            mode="a",
            header=not pd.io.common.file_exists(csv_filename),
            index=False,
        )
        print(f"Datos guardados en {csv_filename} - {df['timestamp'].values[0]}")

    # Esperar X segundos antes de la siguiente recolección
    time.sleep(60)  # Recolecta datos cada 60 segundos (puedes cambiarlo)
