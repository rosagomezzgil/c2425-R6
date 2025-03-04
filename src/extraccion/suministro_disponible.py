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


# Archivo donde se guardarán los datos
csv_filename = "C:\\Users\\vegag\\SEGUNDO\\pd1\\historico_market_data.csv"

# Bucle para recolectar datos cada X segundos
while True:
    df = get_market_data("BTCUSDT")

    if df is not None:
        # Guardar los datos en CSV sin borrar los anteriores
        df.to_csv(csv_filename, mode="a", header=not pd.io.common.file_exists(csv_filename), index=False)
        print(f"✅ Datos guardados en {csv_filename} - {df['timestamp'].values[0]}")

    # Esperar X segundos antes de la siguiente recolección
    time.sleep(60)  # Recolecta datos cada 60 segundos (puedes cambiarlo)