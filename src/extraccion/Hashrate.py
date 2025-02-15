import requests
import pandas as pd

# URL de la API de Blockchain.com para el hashrate
url = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json"

# Hacer la solicitud GET a la API
response = requests.get(url)
data = response.json()

# Extraer fechas y valores del hashrate
hashrate_data = pd.DataFrame(data['values'])
hashrate_data['x'] = pd.to_datetime(hashrate_data['x'], unit='s')  # Convertir timestamp a fecha
hashrate_data.rename(columns={'x': 'timestamp', 'y': 'Hashrate'}, inplace=True)

# Mostrar los primeros datos
print(hashrate_data.head())

# Guardar el conjunto de datos
output_path = r"../../data/datos_hashrate.csv"
hashrate_data.to_csv(output_path, index=False)