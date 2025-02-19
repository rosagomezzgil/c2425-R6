import pandas as pd

df = pd.read_csv(r"../../data/datos_coingecko.csv")

print(df.info())  # Ver tipos de datos y valores nulos
print(df.describe())  # Estadísticas básicas
print(df.head())  # Primeras filas

#convertimos la columna timestamp a datetime
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date


output_path = r"../../data/datos_coingecko_limpios.csv"
df.to_csv(output_path, encoding = "utf-8-sig", index=False)





