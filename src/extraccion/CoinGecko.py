#DataFrame con precio, market cap y volumen del ultimo anno del bitcoin(mas es de pago)
#CoinGecko
import requests
import pandas as pd

def get_coingecko_data(symbol="bitcoin"):
#def get_coingecko_data(symbols=["bitcoin", "ethereum", "cardano"]):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": "usd", "days": 365}

    response = requests.get(url, params=params)
    assert response.status_code == 200

    data = response.json()

    # Convertir datos a DataFrame
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["market_cap"] = [x[1] for x in data["market_caps"]]
    df["volume"] = [x[1] for x in data["total_volumes"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


# Ejecutar la función y verificar el resultado
coingecko_df_bitcoin = get_coingecko_data()
coingecko_df_bitcoin = coingecko_df_bitcoin.iloc[:-1]
print(coingecko_df_bitcoin.head())

# Guardar el conjunto de datos
output_path = r"../../data/datos_bitcoin_coingecko.csv"
coingecko_df_bitcoin.to_csv(output_path, index=False)

