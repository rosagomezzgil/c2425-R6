#DataFrame con el volumen de búsquedas de "Bitcoin" en Google.
#Google Trends
from pytrends.request import TrendReq
from typing import List, Optional

def get_google_trends(
    keywords: List[str] = ["bitcoin"],
    timeframe: str = "today 5-y",
    geo: str = "",
    output_path: Optional[str] = None
):
    """
    Obtiene datos de tendencias de Google Trends.

    Parámetros:
    - keywords (List[str]): Lista de palabras clave a consultar.
    - timeframe (str): Período de tiempo ('today 5-y', '2020-01-01 2024-01-01', etc.).
    - geo (str): Código de país ('US' para EE.UU., 'ES' para España, vacío para global).
    - output_path (str, opcional): Ruta donde guardar el CSV.

    Retorna:
    - pd.DataFrame con los datos obtenidos.
    """
    pytrends = TrendReq()
    pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)

    df = pytrends.interest_over_time().reset_index()

    # Renombrar y filtrar columnas
    df.rename(columns={"date": "timestamp"}, inplace=True)
    df = df[["timestamp"] + keywords]

    # TODO: Revisar si es necesario convertir a formato largo si hay varias palabras clave
    # df = df.melt(id_vars=["timestamp"], var_name="keyword", value_name="google_trends")

    # Guardar CSV si se proporciona una ruta
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Datos guardados en: {output_path}")

    return df

if __name__ == '__main__':
    google_trends_df = get_google_trends(
        keywords=["bitcoin"],
        output_path="../../data/datos_google_trends.csv"
    )

    print(google_trends_df.head())
