import tweepy
import pandas as pd
import config  # Importar las credenciales
from typing import Optional


def fetch_tweets(
        query: str,
        max_results: int = 100,
        output_filename: Optional[str] = None):
    """
    Descarga tweets según una consulta dada y los guarda en un CSV.

    :param query: Consulta de búsqueda en Twitter.
    :param max_results: Número máximo de tweets a recuperar.
    :param output_filename: Ruta del archivo CSV donde se guardarán los tweets.
    :return: DataFrame con los tweets descargados.
    """
    # Autenticación
    client = tweepy.Client(bearer_token=config.BEARER_TOKEN)

    # Obtener tweets
    tweets = client.search_recent_tweets(
        query=query,
        tweet_fields=["created_at", "public_metrics", "author_id"],
        max_results=max_results
    )

    # Convertir los tweets en un DataFrame
    data = []
    for tweet in tweets.data:
        data.append([
            tweet.created_at,
            tweet.text,
            tweet.public_metrics["like_count"],
            tweet.public_metrics["retweet_count"]
        ])

    df = pd.DataFrame(data, columns=["Fecha", "Texto", "Likes", "Retweets"])

    # Guardar el conjunto de datos
    if output_filename:
        df.to_csv(output_filename, index=False)
        print(f"Tweets guardados en: {output_filename}")

    return df


if __name__ == "__main__":
    # Parámetros de extracción
    query = "(Bitcoin OR BTC OR #Bitcoin) -is:retweet lang:en"
    max_results = 100
    output_path = "../../../data/tweets_bitcoin.csv"

    df_twitter = fetch_tweets(query=query, max_results=max_results, output_filename=output_path)

    print(df_twitter.shape)
    print(df_twitter.columns)
    print(df_twitter.head())



