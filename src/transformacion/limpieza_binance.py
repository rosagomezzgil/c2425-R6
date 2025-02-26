import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Cargar datos desde un CSV (ajusta la ruta a tu archivo)
    df = pd.read_csv(r"../../data/datos_bitcoin_binance.csv")

    # Ver las primeras filas
    print(df.head())
    #Comprobar el tipo de las variables
    print(df.dtypes)

    #Convertir timestamp a fecha
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convertir a formato fecha
    df.sort_values('timestamp', inplace=True)  # Ordenar por fecha

    # Ver si hay valores nulos
    print(df.isnull().sum())    #No tiene valores nulos

    #annadir columnas: Medias moviles
    # Media Móvil Simple (SMA)
    df['SMA_50'] = df['close_binance'].rolling(window=50).mean()  # 50 días
    df['SMA_200'] = df['close_binance'].rolling(window=200).mean()  # 200 días

    # Media Móvil Exponencial (EMA)
    df['EMA_50'] = df['close_binance'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['close_binance'].ewm(span=200, adjust=False).mean()

    #Graficar las medias moviles para ver si reflejan bien la tendencia
    plt.figure(figsize=(12,6))

        # Graficamos el precio de cierre
    plt.plot(df['timestamp'], df['close_binance'], label="Precio de cierre", color="blue")

        # Graficamos las medias móviles
    plt.plot(df['timestamp'], df['SMA_50'], label="SMA 50", color="orange")
    plt.plot(df['timestamp'], df['SMA_200'], label="SMA 200", color="red")

    plt.title("Bitcoin - Precio y Medias Móviles")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.show()

    df.to_csv("../../data/datos_bitcoin_limpiados_binance.csv", index=False)