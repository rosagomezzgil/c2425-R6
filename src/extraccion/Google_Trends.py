import time
from pytrends.request import TrendReq
from typing import List, Optional
import pandas as pd


def get_google_trends(
    keywords: List[str] = ["bitcoin"],
    timeframe: str = "today 5-y",
    geo: str = "",
    output_path: Optional[str] = None,
):
    pytrends = TrendReq()

    # Pausa antes de hacer la solicitud para evitar bloqueos
    time.sleep(15)

    pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)

    df = pytrends.interest_over_time().reset_index()

    if df.empty:
        print(
            "⚠️ No se recibieron datos. Puede que Google esté bloqueando la solicitud."
        )
        return pd.DataFrame()

    df.rename(columns={"date": "timestamp"}, inplace=True)
    df = df[["timestamp"] + keywords]

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Datos guardados en: {output_path}")

    return df


if __name__ == "__main__":
    google_trends_df = get_google_trends(
        keywords=["bitcoin"],
        timeframe="today 5-y",
        output_path="../../data/extraccion/datos_google_trends.csv",
    )

    if not google_trends_df.empty:
        print(google_trends_df.iloc[0, 0])
        print(google_trends_df.iloc[-1, 0])
        print(google_trends_df.shape)
        print(google_trends_df.head())
