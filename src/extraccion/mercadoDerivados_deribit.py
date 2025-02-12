import requests
import pandas as pd
import time


# Función para obtener datos históricos de Deribit en múltiples solicitudes
def get_deribit_historical_data(instrument="BTC-PERPETUAL", days=365, resolution="60"):
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"

    # Tiempo actual en milisegundos
    end_timestamp = int(time.time() * 1000)
    # Tiempo de inicio (hace un año)
    start_timestamp = end_timestamp - (days * 24 * 60 * 60 * 1000)

    all_data = []

    # Iterar en bloques de 24 horas
    interval_ms = 24 * 60 * 60 * 1000  # 24 horas en milisegundos
    current_start = start_timestamp

    while current_start < end_timestamp:
        current_end = min(current_start + interval_ms, end_timestamp)

        params = {
            "instrument_name": instrument,
            "start_timestamp": current_start,
            "end_timestamp": current_end,
            "resolution": resolution  # Resolución: "1", "5", "15", "60", etc.
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "result" in data and data["result"]:
            df = pd.DataFrame(data["result"])
            df["timestamp"] = pd.to_datetime(df["ticks"], unit="ms")  # Convertir timestamp a fecha legible
            df.set_index("timestamp", inplace=True)
            all_data.append(df[["open", "high", "low", "close", "volume"]])
        else:
            print("Error al obtener datos en el rango:", data)

        # Avanzar al siguiente intervalo de 24 horas
        current_start += interval_ms
        time.sleep(0.5)  # Evitar ser bloqueado por la API

    # Concatenar todos los datos obtenidos
    if all_data:
        full_df = pd.concat(all_data)
        return full_df.sort_index()  # Ordenar por fecha
    else:
        print("No se obtuvieron datos.")
        return None


# Llamada a la función (últimos 365 días, velas de 1 hora)
df = get_deribit_historical_data(days=365, resolution="60")

if df is not None:
    df.to_csv("C:\\Users\\vegag\\SEGUNDO\\pd1\\mercadoDerivados_derivit.csv", encoding="utf-8-sig",
              index=True)  # Cambia el nombre del archivo si lo deseas

# Mostrar las primeras filas
df.head()