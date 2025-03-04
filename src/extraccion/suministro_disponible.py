import requests
import pandas as pd
import os


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
            "quoteVolume": float(data["quoteVolume"])
        }
        return pd.DataFrame([market_data])
    else:
        print("Error al obtener datos:", response.status_code)
        return None

if __name__ == '__main__':
    # Archivo donde se guardarán los datos
    csv_filename = r'../../data/extraccion/historico_market_data.csv'

<<<<<<< Updated upstream
    # Bucle para recolectar datos cada X segundos
    while True:
        df = get_market_data("BTCUSDT")

        if df is not None:
            # Guardar los datos en CSV sin borrar los anteriores
            df.to_csv(csv_filename, mode="a", header=not pd.io.common.file_exists(csv_filename), index=False)
            print(f"✅ Datos guardados en {csv_filename} - {df['timestamp'].values[0]}")

        # Esperar X segundos antes de la siguiente recolección
        time.sleep(60)  # Recolecta datos cada 60 segundos (puedes cambiarlo)
=======
# Carpeta donde se guardarán los datos
data_folder = "C:\\Users\\vegag\\SEGUNDO\\pd1\\historico_market_data"

# Crear la carpeta si no existe
os.makedirs(data_folder, exist_ok=True)

# Obtener los datos del día actual
df = get_market_data("BTCUSDT")

if df is not None:
    # Obtener la fecha actual en formato YYYY-MM-DD
    today_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Nombre del archivo CSV basado en la fecha
    csv_filename = os.path.join(data_folder, f"market_data_{today_date}.csv")

    # Guardar los datos en un archivo único para el día
    df.to_csv(csv_filename, mode="a", header=not os.path.exists(csv_filename), index=False)

    print(f"✅ Datos guardados en {csv_filename} - {df['timestamp'].values[0]}")

>>>>>>> Stashed changes
