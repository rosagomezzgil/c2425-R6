import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv(r"../../data/datos_bitcoin_google_trends.csv")
    df = df.reset_index(drop=True)
    df = df.drop(columns=[df.columns[0]])
    # Guardar el conjunto de datos
    output_path = r"../../data/limpios_google_trends.csv"
    df.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df.head())
