import pandas as pd

# Leer el archivo CSV en un DataFrame
df = pd.read_csv(r"../../data/bitget_scrapping.csv")

# Mostrar las primeras filas del DataFrame
print("\n✅ Primeras filas del archivo CSV:")
print(df.head())

# Mostrar información general del DataFrame
print("\n🔍 Información del dataset:")
print(df.info())

# Mostrar estadísticas básicas de los datos
print("\n📊 Resumen estadístico:")
print(df.describe())


df = df.drop_duplicates()

df = df.dropna()

df.columns = df.columns.str.lower()

# Seleccionar todas las columnas excepto 'fecha'
numeric_cols = df.columns.difference(["fecha"])  # Obtiene todas menos 'fecha'

# Convertir a float
df[numeric_cols] = df[numeric_cols].replace('[€,B]', '', regex=True).astype(float)

# Mostrar resultado
print(df.head())
print(df.info())

### 🛠️ 1️⃣ LIMPIEZA DE FORMATO DE DATOS ###

# Convertir la columna 'Fecha' a datetime
df["fecha"] = pd.to_datetime(df["fecha"], format="%Y-%m-%d")

# Función para convertir valores monetarios en float
def convertir_euros(valor):
    return float(valor.replace("€", "").replace(",", "")) if isinstance(valor, str) else valor

# Aplicar la conversión a todas las columnas numéricas
columnas_monetarias = ["abrir*", "máximo", "mínimo", "cerrar**"]
for col in columnas_monetarias:
    df[col] = df[col].apply(convertir_euros)

# Función para convertir volumen (B de Billones a número real)
def convertir_volumen(vol):
    if isinstance(vol, str):  # Solo aplicar reemplazos si vol es un string
        if "B" in vol:
            return float(vol.replace("€", "").replace("B", "")) * 1e9  # Convertir a número real
        return float(vol.replace("€", "").replace(",", ""))
    return vol  # Si ya es float, devolverlo tal cual

df["volumen"] = df["volumen"].apply(convertir_volumen)


### 🛠️ 2️⃣ DETECCIÓN Y TRATAMIENTO DE VALORES NULOS ###
print("\n🔍 Valores nulos en el dataset:")
print(df.isnull().sum())

# Si hay valores nulos en precios o volumen, se imputan con la media
df.fillna(df.mean(), inplace=True)

### 🛠️ 3️⃣ DETECCIÓN DE OUTLIERS ###
# Definir una función para detectar outliers con el método IQR
def detectar_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = data[(data[col] < limite_inferior) | (data[col] > limite_superior)]
    return outliers

print("\n📌 Outliers detectados en el dataset:")
for col in columnas_monetarias + ["volumen"]:
    outliers = detectar_outliers_iqr(df, col)
    if not outliers.empty:
        print(f"\n🔴 Outliers en {col}:")
        print(outliers)

#no vamos a eliminar outliers ya que pueden ser utiles para la prediccion, solamente los detectamos

### 🛠️ 4️⃣ RESULTADO FINAL ###
print("\n✅ Dataset limpio y listo para su análisis:")
print(df.head())

df.to_csv("../../data/bitget_scrapping.csv", index=False)
