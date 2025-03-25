import tweepy
import pandas as pd
import argparse
import config  # Importar las credenciales
from typing import Optional

def fetch_tweets(
    query: str, max_results: int = 100, output_filename: Optional[str] = None
):
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
        max_results=max_results,
    )

    # Convertir los tweets en un DataFrame
    data = []
    for tweet in tweets.data:
        data.append(
            [
                tweet.created_at,
                tweet.text,
                tweet.public_metrics["like_count"],
                tweet.public_metrics["retweet_count"],
            ]
        )

    df = pd.DataFrame(data, columns=["Fecha", "Texto", "Likes", "Retweets"])

    # Guardar el conjunto de datos
    if output_filename:
        df.to_csv(output_filename, index=False)
        print(f"Tweets guardados en: {output_filename}")

    return df

def main():
    parser = argparse.ArgumentParser("Descargar tweets según una consulta dada y los guarda en un CSV")
    parser.add_argument(
        "--query", type=str, default="(Bitcoin OR BTC OR #Bitcoin) -is:retweet lang:en",
        help="Consulta de búsqueda en Twitter"
    )
    parser.add_argument(
        "--max_results", type=int, default=100, help="Número máximo de tweets a recuperar"
    )
    parser.add_argument(
        "--output_path", type=str, default="../../../data/extraccion/tweets_bitcoin.csv",
        help="Ruta del archivo CSV donde se guardarán los tweets"
    )

    # Parsear los argumentos
    args = parser.parse_args()

    # Llamar a la funcion con los argumentos parseados
    df_twitter = fetch_tweets(
        query=args.query,
        max_results=args.max_results,
        output_filename=args.output_path
    )

    print(df_twitter.shape)
    print(df_twitter.columns)
    print(df_twitter.head())

if __name__ == "__main__":
    main()