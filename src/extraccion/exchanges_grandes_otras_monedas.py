import requests
import pandas as pd
from typing import Optional, List

# Función para obtener Exchange Info
def get_exchange_info(base_asset: Optional[str] = "BTC", quote_asset: Optional[str] = None, output_path: Optional[str] = None):
    """
    Obtiene información de pares de trading en Binance.

    :param base_asset: (str, opcional) Filtrar pares donde este activo sea la base.
    :param quote_asset: (str, opcional) Filtrar pares donde este activo sea la cotización.
    :param output_path: (str, opcional) Ruta donde guardar el CSV.

    :return: pd.DataFrame con los pares de trading filtrados.
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"

    try:
        response = requests.get(url)
        response.raise_for_status()

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
            for s in symbols
            if (not base_asset or s["baseAsset"] == base_asset) and (not quote_asset or s["quoteAsset"] == quote_asset)
        ]

        df = pd.DataFrame(btc_pairs)

        if output_path:
            df.to_csv(output_path, index=True)
            print(f"Datos guardados en: {output_path}")

        return df

    except requests.exceptions.RequestException as e:
        print("Error al obtener Exchange Info:", e)
        return None


# Función para obtener Market Data de un par de trading
def get_market_data(symbols: List[str], output_path: Optional[str] = None):
    """
    Obtiene datos de mercado de múltiples pares de trading.

    :param symbols: (List[str]) Lista de pares de trading.
    :param output_path: (str, opcional) Ruta donde guardar el CSV.

    :return: pd.DataFrame con los datos de mercado.
    """
    market_data_list = []

    for symbol in symbols:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        try:
            response = requests.get(url)
            response.raise_for_status()

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
            market_data_list.append(market_data)

        except requests.exceptions.RequestException as e:
            print(f"Error al obtener datos de {symbol}: {e}")

    df = pd.DataFrame(market_data_list)

    if output_path:
        df.to_csv(output_path, encoding="utf-8-sig", index=False)
        print(f"Datos guardados en: {output_path}")

    return df


if __name__ == '__main__':
    df_btc_pairs = get_exchange_info(base_asset="BTC")
    print(df_btc_pairs.head())
    print(df_btc_pairs.shape)

    df_market_data = get_market_data(df_btc_pairs["symbol"].tolist(), output_path="../../data/btc_market_data.csv")
    print(df_market_data.head())
    print(df_market_data.shape)