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

#Sp500

# Descargar datos del S&P 500
sp500 = yf.download('^GSPC', start="2010-01-01", end="2025-02-17", interval="1d")

# Descargar datos de Bitcoin
btc = yf.download('BTC-USD', start="2010-01-01", end="2025-02-17", interval="1d")

#Renombrar las columnas
sp500.columns = ['Open_sp500', 'High_sp500', 'Low_sp500', 'Close_sp500', 'Volume_sp500']

# Convertir el índice de fecha en una columna llamada timestamp
sp500 = sp500.reset_index().rename(columns={'Date': 'timestamp'})

print(sp500.head())

# Guardar el conjunto de datos
output_path = r"../../data/datos_bitcoin_Sp500.csv"
dxy.to_csv(output_path, index=False)

#Si encuentras una correlación alta, se puede usar el S&P 500 como indicador adelantado para prever movimientos de Bitcoin.


#Nasdaq

# Definir el período de tiempo
start_date = "2015-01-01"
end_date = "2025-02-17"

# Descargar datos históricos
nasdaq = yf.download('^IXIC', start=start_date, end=end_date, interval="1d")

#Renombrar las columnas
nasdaq.columns = ['Open_nasdaq', 'High_nasdaq', 'Low_nasdaq', 'Close_nasdaq', 'Volume_nasdaq']

# Convertir el índice de fecha en una columna llamada timestamp
nasdaq = nasdaq.reset_index().rename(columns={'Date': 'timestamp'})

print(nasdaq.head())

# Guardar el conjunto de datos
output_path = r"../../data/datos_bitcoin_Nasdaq.csv"
dxy.to_csv(output_path, index=False)

#Si encuentras una correlación alta, se puede usar el nasdaq como indicador adelantado para prever movimientos de Bitcoin.
#Tiene subidas y bajadas mas bruscas que el sp500 y mas afines al BTC

#ETFs

# Definir los ETFs de Bitcoin que queremos analizar
etfs = ["BITO", "IBIT", "ARKB"]

# Descargar los datos históricos y almacenarlos correctamente
data = {}
for etf in etfs:
    data[etf] = yf.download(etf, start="2021-01-01", end="2025-01-01", interval="1d")['Close']

# Convertir el diccionario a un DataFrame
df_ETFs = pd.concat(data, names=["Ticker"], axis=1).reset_index()

# Renombrar las columnas para claridad
df_ETFs.columns = ["Date"] + list(data.keys())

# Mostrar el DataFrame
print(df_ETFs.head())

# Guardar el conjunto de datos
output_path = r"../../data/datos_bitcoin_EFTs.csv"
dxy.to_csv(output_path, index=False)