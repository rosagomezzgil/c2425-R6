import requests
import pandas as pd
import matplotlib.pyplot as plt

# URL de la API de Blockchain.com para el hashrate
url = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json"

# Hacer la solicitud GET a la API
response = requests.get(url)
data = response.json()

# Extraer fechas y valores del hashrate
hashrate_data = pd.DataFrame(data['values'])
hashrate_data['x'] = pd.to_datetime(hashrate_data['x'], unit='s')  # Convertir timestamp a fecha
hashrate_data.rename(columns={'x': 'Date', 'y': 'Hashrate'}, inplace=True)

# Mostrar los primeros datos
hashrate_data