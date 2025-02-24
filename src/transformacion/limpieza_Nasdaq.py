import pandas as pd

if __name__ == '__main__':
    # Cargar datos desde un CSV (ajusta la ruta a tu archivo)
    nasdaq = pd.read_csv(r"../../data/datos_bitcoin_Nasdaq.csv")

    # Verificar estructura
    print(nasdaq.info())
    print(nasdaq.describe())

    # Variaciones porcentuales diarias
    nasdaq["daily_return"] = (nasdaq["Close"] - nasdaq["Open"]) / nasdaq["Open"] * 100

    # Volatilidad diaria
    nasdaq["volatility"] = (nasdaq["High"] - nasdaq["Low"]) / nasdaq["Open"]

    # Media móvil de 7 y 30 días
    nasdaq["sma_7"] = nasdaq["Close"].rolling(window=7).mean()
    nasdaq["sma_30"] = nasdaq["Close"].rolling(window=30).mean()

