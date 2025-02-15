import tweepy
import pandas as pd
import config  # Importar las credenciales

# Autenticación
client = tweepy.Client(bearer_token=config.BEARER_TOKEN)

# Definir búsqueda (Ejemplo: tweets sobre Bitcoin en inglés)
query = "(Bitcoin OR BTC OR #Bitcoin) -is:retweet lang:en"

# Obtener tweets recientes (máximo 100)
tweets = client.search_recent_tweets(
    query=query,
    tweet_fields=["created_at", "public_metrics", "author_id", "context_annotations"],
    user_fields=["username", "verified", "public_metrics"],
    expansions=["author_id"],
    max_results=100
)

# Convertir los tweets en un DataFrame
data = []
for tweet in tweets.data:
    data.append([tweet.created_at, tweet.text, tweet.public_metrics["like_count"], tweet.public_metrics["retweet_count"]])

df = pd.DataFrame(data, columns=["Fecha", "Texto", "Likes", "Retweets"])
print(df.head())

# Guardar el conjunto de datos
output_path = r"../../../data/tweets_bitcoin.csv"
df.to_csv(output_path, index=False)


