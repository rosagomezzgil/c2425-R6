#Macroeconomicos
#Índice del dólar (DXY)
#Yahoo Finance

import yfinance as yf
import pandas as pd

# Descargar los datos del Índice del Dólar (DXY) desde Yahoo Finance
dxy = yf.download("DX-Y.NYB", period="5y", interval="1d")

# Asegurar que la fecha sea una columna normal y no un índice
dxy = dxy.reset_index()

# Si las columnas tienen un MultiIndex, lo corregimos
if isinstance(dxy.columns, pd.MultiIndex):
    dxy.columns = dxy.columns.get_level_values(0)  # Tomamos solo el primer nivel de nombres

# Verificar nombres de columnas antes de renombrar
print("Nombres de columnas:", dxy.columns)

# Renombrar correctamente las columnas
dxy = dxy.rename(columns={"Date": "timestamp", "Close": "dxy"})

# Seleccionar solo las columnas necesarias
dxy = dxy[["timestamp", "dxy"]]
print(dxy.head())

# Guardar el conjunto de datos
output_path = r"../../data/datos_bitcoin_dxy.csv"
dxy.to_csv(output_path, index=False)

#Si DXY sube, significa que el dólar es más fuerte y los inversores podrían vender BTC.
#Si DXY baja, BTC podría ser más atractivo como alternativa al dólar.