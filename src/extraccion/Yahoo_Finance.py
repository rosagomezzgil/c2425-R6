import yfinance as yf
import pandas as pd
import os
import time

def download_data(ticker, start_date, end_date, interval, columns_rename=None, output_filename=None, max_retries=5):
    """
    Descarga datos de Yahoo Finance, maneja errores y los guarda en un CSV.

    :param ticker: Símbolo del activo a descargar de Yahoo Finance.
    :param start_date: Fecha de inicio en formato "YYYY-MM-DD".
    :param end_date: Fecha de fin en formato "YYYY-MM-DD".
    :param interval: Intervalo de tiempo entre datos (por ejemplo, "1d" para diario).
    :param columns_rename: Diccionario opcional para renombrar columnas.
    :param output_filename: Ruta del archivo CSV donde se guardarán los datos.
    :param max_retries: Número máximo de intentos en caso de error de límite de solicitudes.
    :return: DataFrame con los datos descargados o None si falla.
    """
    retries = 0
    while retries < max_retries:
        try:
            # Descargar datos
            df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

            # Validar si el DataFrame está vacío
            if df.empty:
                print(f"Advertencia: No se encontraron datos para {ticker}")
                return None

            # Convertir índice de fecha en columna normal
            df = df.reset_index()

            # Renombrar columnas si es necesario
            if columns_rename:
                df = df.rename(columns=columns_rename)

            # Seleccionar solo las columnas necesarias si se han renombrado
            if 'timestamp' in df.columns and 'dxy' in df.columns:
                df = df[['timestamp', 'dxy']]

            # Guardar el conjunto de datos
            if output_filename:
                df.to_csv(output_filename, index=False)
                print(f"Datos guardados en: {output_filename}")

            return df
        except Exception as e:
            retries += 1
            wait_time = 10 * retries
            print(f"Error en {ticker}: {e}. Reintentando en {wait_time} segundos...")
            time.sleep(wait_time)

    print(f"Error crítico: No se pudo descargar {ticker} después de {max_retries} intentos.")
    return None

if __name__ == '__main__':
    # Parámetros
    start_date = "2022-01-01"
    end_date = "2025-02-17"
    interval = "1d"
    output_dir = "../../data"
    os.makedirs(output_dir, exist_ok=True)

    # Índice del Dólar (DXY)
    dxy = download_data(
        ticker="DX-Y.NYB",
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        columns_rename={"Date": "timestamp", "Close": "dxy"},
        output_filename=os.path.join(output_dir, "datos_dxy.csv")
    )

    # S&P 500
    sp500 = download_data(
        ticker="^GSPC",
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        columns_rename={"Date": "timestamp"},
        output_filename=os.path.join(output_dir, "datos_sp500.csv")
    )

    # Nasdaq
    nasdaq = download_data(
        ticker="^IXIC",
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        columns_rename={"Date": "timestamp"},
        output_filename=os.path.join(output_dir, "datos_nasdaq.csv")
    )

    """
    # ETFs de Bitcoin
    etfs = ["BITO", "IBIT", "ARKB"]
    data = {}

    for etf in etfs:
        data[etf] = download_data(
            ticker=etf,
            start_date=start_date,
            end_date=end_date,  # Se unifica la fecha de fin con los demás
            interval=interval
        )

    # Concatenar datos y guardar CSV
    df_ETFs = pd.concat(data, axis=1).reset_index()

    if not df_ETFs.empty:
        df_ETFs.to_csv(os.path.join(output_dir, "datos_etfs.csv"), index=False)
        print(f"Datos de ETFs guardados en: {os.path.join(output_dir, 'datos_etfs.csv')}")
    else:
        print("Advertencia: No se guardaron datos de ETFs debido a falta de datos.")

"""