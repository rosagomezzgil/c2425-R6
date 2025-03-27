import requests
import pandas as pd
import argparse
from typing import Optional


def get_bitcoin_hashrate(
    timespan: str = "all",
    rolling_average: str = "24hours",
    output_path: Optional[str] = None,
):
    """
    Obtiene datos del hashrate de Bitcoin desde Blockchain.com.

    :param timespan: (str) Rango de tiempo (ej. "30days", "1year", "all").
    :param rolling_average: (str) Promedio móvil (ej. "8hours", "24hours").
    :param output_path: (str, opcional) Ruta para guardar los datos en CSV.

    :return: pd.DataFrame con la información del hashrate.
    """

    url = f"https://api.blockchain.info/charts/hash-rate?timespan={timespan}&rollingAverage={rolling_average}&format=json"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Verifica errores HTTP

        data = response.json()
        df = pd.DataFrame(data["values"])

        # Convertir timestamp a formato de fecha
        df["x"] = pd.to_datetime(df["x"], unit="s")
        df.rename(columns={"x": "timestamp", "y": "hashrate"}, inplace=True)

        # Guardar CSV si se proporciona una ruta
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Datos guardados en: {output_path}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        "Obtener datos del hashrate de Bitcoin desde Blockchain.com"
    )
    parser.add_argument(
        "--timespan",
        type=str,
        default="all",
        help="Rango de tiempo (ej. 30days, 1year, all)",
    )
    parser.add_argument(
        "--rolling_average",
        type=str,
        default="24hours",
        help="Promedio móvil (ej. 8hours, 24hours)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=r"../../data/extraccion/datos_hashrate.csv",
        help="Ruta para guardar los datos en CSV",
    )

    # Parsear los argumentos
    args = parser.parse_args()

    # Llamar a la funcion con los argumentos parseados
    hashrate_df = get_bitcoin_hashrate(
        timespan=args.timespan,
        rolling_average=args.rolling_average,
        output_path=args.output_path,
    )

    print(hashrate_df.shape)
    print(hashrate_df.head())


if __name__ == "__main__":
    main()
