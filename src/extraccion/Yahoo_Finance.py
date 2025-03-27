# Ejecutar en Google Colab, porque en local da error de Too Many Requests

import yfinance as yf
import argparse
import os
import time
from typing import Optional


def download_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    columns_rename: Optional[dict] = None,
    output_filename: Optional[str] = None,
    max_retries: int = 5,
):
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

    print(
        f"Error crítico: No se pudo descargar {ticker} después de {max_retries} intentos."
    )
    return None


def main():
    start_date = "2022-01-01"
    end_date = "2025-02-17"

    parser = argparse.ArgumentParser(description="Descargar datos de Yahoo Finance.")
    parser.add_argument(
        "--start_date",
        type=str,
        default=start_date,
        help="Fecha de inicio en formato YYYY-MM-DD",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=end_date,
        help="Fecha de fin en formato YYYY-MM-DD",
    )
    parser.add_argument(
        "--interval", type=str, default="1d", help="Intervalo de datos (ej. '1d', '1h')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../data/extraccion",
        help="Directorio de salida para los archivos CSV",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    assets = {
        "DX-Y.NYB": "datos_dxy.csv",
        "^GSPC": "datos_sp500.csv",
        "^IXIC": "datos_nasdaq.csv",
    }

    for ticker, filename in assets.items():
        download_data(
            ticker=ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            columns_rename={"Date": "timestamp"},
            output_filename=os.path.join(args.output_dir, filename),
        )


if __name__ == "__main__":
    main()
