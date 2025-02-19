import requests
import pandas as pd


# Función para obtener Exchange Info
def get_exchange_info():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        symbols = data["symbols"]

        # Filtrar solo los pares que contienen "BTC"
        btc_pairs = [
            {
                "symbol": s["symbol"],
                "baseAsset": s["baseAsset"],
                "quoteAsset": s["quoteAsset"],
                "status": s["status"]
            }
            for s in symbols if "BTC" in [s["baseAsset"], s["quoteAsset"]]
        ]

        return pd.DataFrame(btc_pairs)
    else:
        print("Error al obtener Exchange Info:", response.status_code)
        return None


# Función para obtener Market Data de un par de trading
def get_market_data(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        market_data = {
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
        print("Error al obtener Market Data:", response.status_code)
        return None


# Obtener pares de trading relacionados con BTC
df_btc_pairs = get_exchange_info()

# Obtener datos de mercado de los pares relacionados con BTC (por ejemplo, BTC/USDT, BTC/ETH)
market_data_list = []
for symbol in df_btc_pairs["symbol"]:  # Limitar a los primeros 10 pares (puedes cambiar el número)
    df = get_market_data(symbol)
    if df is not None:
        market_data_list.append(df)

# Combinar los datos en un solo DataFrame
df_market_data = pd.concat(market_data_list, ignore_index=True)

df_market_data.to_csv("C:\\Users\\vegag\\SEGUNDO\\pd1\\suministro_disponible.csv", encoding="utf-8-sig", index=False)

# Mostrar los datos de mercado
print("Datos de mercado de los pares relacionados con BTC:")
print(df_market_data)