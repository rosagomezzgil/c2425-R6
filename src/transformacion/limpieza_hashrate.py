import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if __name__ == '__main__':
    # Cargar datos desde un CSV (ajusta la ruta a tu archivo)
    df = pd.read_csv(r"../../data/datos_hashrate.csv")

    # Ver las primeras filas
    print(df.head())
    #Comprobar el tipo de las variables
    print(df.dtypes)

    # Convertir timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Crear un rango de fechas completo con frecuencia diaria
    df_complete = pd.DataFrame(
        {'timestamp': pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')})

    # Unir con el dataset original
    df_merged = pd.merge(df_complete, df, on='timestamp', how='left')

    # Contar cuántos días faltaban antes de la interpolación
    missing_days = df_complete.shape[0] - df.shape[0] 

    # Ver cuántos días consecutivos sin datos había antes de la interpolación
    df['diff'] = df['timestamp'].diff().dt.days  # Diferencia en días entre fechas
    max_gap = df['diff'].max()  # Mayor hueco de días sin datos

    print(f"Total de fechas faltantes: {missing_days}")
    print(f"Máximo intervalo sin datos: {max_gap} días")

    # Aplicar interpolación lineal
    df_merged['hashrate'] = df_merged['hashrate'].interpolate(method='linear')

    # Calcular medias móviles con ventanas de 50 y 200 días
    df_merged['SMA_50'] = df_merged['hashrate'].rolling(window=50).mean()  # Media móvil simple 50 días
    df_merged['SMA_200'] = df_merged['hashrate'].rolling(window=200).mean()  # Media móvil simple 200 días
    df_merged['EMA_50'] = df_merged['hashrate'].ewm(span=50, adjust=False).mean()  # Media móvil exponencial 50 días
    df_merged['EMA_200'] = df_merged['hashrate'].ewm(span=200, adjust=False).mean()  # Media móvil exponencial 200 días

    # Graficar las medias móviles
    plt.figure(figsize=(12, 6))
    plt.plot(df_merged['timestamp'], df_merged['hashrate'], label='Hashrate Original', color='lightgray', alpha=0.5)
    plt.plot(df_merged['timestamp'], df_merged['SMA_50'], label='SMA 50 días', color='blue', alpha=0.7)
    plt.plot(df_merged['timestamp'], df_merged['SMA_200'], label='SMA 200 días', color='red', alpha=0.7)
    plt.plot(df_merged['timestamp'], df_merged['EMA_50'], label='EMA 50 días', color='green', linestyle='dashed',
             alpha=0.7)
    plt.plot(df_merged['timestamp'], df_merged['EMA_200'], label='EMA 200 días', color='purple', linestyle='dashed',
             alpha=0.7)
    plt.xlabel('Fecha')
    plt.ylabel('Hashrate')
    plt.title('Hashrate con Medias Móviles (50 y 200 días)')
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.show()

    # Ver si hay valores nulos
    print(df.isnull().sum())  # No tiene valores nulos

    df.to_csv("../../data/datos_hashrate_limpiados_binance.csv", index=False)