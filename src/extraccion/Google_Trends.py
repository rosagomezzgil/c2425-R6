#DataFrame con el volumen de búsquedas de "Bitcoin" en Google.
#Google Trends
from pytrends.request import TrendReq

def get_google_trends(keyword="bitcoin"):
    pytrends = TrendReq()
    pytrends.build_payload([keyword], timeframe="today 5-y")
    df = pytrends.interest_over_time()
    df = df.reset_index()
    df.rename(columns={keyword: "google_trends"}, inplace=True)
    df = df[["date", "google_trends"]]
    df.rename(columns={"date": "timestamp"}, inplace=True)

    return df

google_trends_df = get_google_trends()
print(google_trends_df.head)

# Guardar el conjunto de datos
output_path = r"../../data/datos_google_trends.csv"
google_trends_df.to_csv(output_path, index=False)
