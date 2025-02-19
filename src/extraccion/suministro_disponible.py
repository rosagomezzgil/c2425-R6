import os
import time
import requests
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

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

# Autenticación con Google Drive
gauth = GoogleAuth()

# Intenta cargar las credenciales guardadas o solicita nueva autenticación
gauth.LoadCredentialsFile("credentials.json")

if not gauth.credentials:
    gauth.LocalWebserverAuth()  # Abre el navegador para la autenticación
elif gauth.access_token_expired:
    gauth.Refresh()  # Actualiza las credenciales si han expirado
else:
    gauth.Authorize()  # Autoriza si las credenciales están disponibles

# Guarda las credenciales para usarlas en el futuro
gauth.SaveCredentialsFile("credentials.json")

# Crea un objeto GoogleDrive con la autenticación
drive = GoogleDrive(gauth)

# ID de la carpeta de Google Drive donde quieres guardar el archivo
folder_id = '1rx4XvYLFwyn86TofruAenmslBwZ_eMg2'  # Usa el ID de tu carpeta de Google Drive

# Archivo CSV local
csv_filename = "suministro_disponible.csv"

# Bucle para recolectar datos cada X segundos
while True:
    df = get_market_data("BTCUSDT")

    if df is not None:
        # Guardar los datos en CSV localmente
        df.to_csv(csv_filename, mode="a", header=not pd.io.common.file_exists(csv_filename), index=False)
        print(f"✅ Datos guardados en {csv_filename} - {df['timestamp'].values[0]}")

        # Subir el archivo CSV a Google Drive
        file_drive = drive.CreateFile({"title": os.path.basename(csv_filename), "parents": [{"id": folder_id}]})
        file_drive.SetContentFile(csv_filename)
        file_drive.Upload()  # Sube el archivo
        print(f"📤 Archivo subido a Google Drive: {file_drive['title']}")

    # Esperar X segundos antes de la siguiente recolección
    time.sleep(60)  # Recolecta datos cada 60 segundos (puedes cambiarlo)
