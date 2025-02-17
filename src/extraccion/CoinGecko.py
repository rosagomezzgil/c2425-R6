#DataFrame con precio, market cap y volumen del ultimo anno del bitcoin(mas es de pago)
#CoinGecko
import requests
import pandas as pd
from typing import Optional

def get_coingecko_data(
    symbol: str = "bitcoin",
    vs_currency: str = "usd",
    days: int = 365,
    output_path: Optional[str] = None
):
    """
    Obtiene datos históricos de criptomonedas desde CoinGecko.

    Parámetros:
    - symbol (str): criptomoneda a consultar (ej. ['bitcoin', 'ethereum']).
    - vs_currency (str): Moneda base ('usd', 'eur', etc.).
    - days (int): Número de días de datos a obtener.
    - output_path (str, opcional): Ruta donde guardar el CSV.

    Retorna:
    - pd.DataFrame con los datos obtenidos.
    """

    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}

    response = requests.get(url, params=params)
    response.raise_for_status() # Lanza un error si la solicitud falla
    data = response.json()

    # Convertir datos a DataFrame
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["market_cap"] = [x[1] for x in data["market_caps"]]
    df["volume"] = [x[1] for x in data["total_volumes"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Guardar CSV si se proporciona una ruta
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Datos guardados en: {output_path}")

    return df


if __name__ == '__main__':
    coingecko_df = get_coingecko_data(
        symbol="bitcoin",
        vs_currency="usd",
        days=365,
        output_path="../../data/datos_coingecko.csv"
    )

    print(coingecko_df.head())

