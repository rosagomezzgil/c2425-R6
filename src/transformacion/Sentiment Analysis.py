import pandas as pd
from textblob import TextBlob


def get_sentiment(text):
    """Analiza el sentimiento de un texto y lo clasifica."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positivo"
    elif polarity < 0:
        return "Negativo"
    else:
        return "Neutral"


def analyze_tweets(file_path):
    """Carga el archivo CSV, analiza el sentimiento de los tweets y muestra estadísticas."""
    # Cargar datos
    df = pd.read_csv(file_path)

    # Asegurar que la columna de texto es string
    df["Texto"] = df["Texto"].astype(str)

    # Aplicar análisis de sentimiento
    df["Sentimiento"] = df["Texto"].apply(get_sentiment)

    # Mostrar resultados
    print(df["Sentimiento"].value_counts())
    return df

if __name__ == '__main__':
    # Ruta del archivo (modificar según sea necesario)
    file_path = r"../../data/tweets_bitcoin.csv"
    df_resultado = analyze_tweets(file_path)
    print(df_resultado.head())

    # Guardar el conjunto de datos
    output_path = r"../../data/sentiment_analysis.csv"
    df_resultado.to_csv(output_path, index=False)