import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Función para cargar datos desde un archivo Parquet
def cargar_datos(ruta_parquet):
    df = pd.read_parquet(ruta_parquet)
    return df


# Función para preprocesar los datos
def preprocesar_datos(df, variables_objetivo=['pct_cambio_30d']):
    df = df.drop(columns=['tendencia_futura_7d', 'pct_cambio_7d', 'tendencia_futura_30d'], errors='ignore')
    X = df.drop(columns=['pct_cambio_30dd'])
    y = df['pct_cambio_30d']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
    df_scaled = pd.concat([df_scaled, y], axis=1)  # Agregar variables objetivo sin escalar

    return df_scaled, scaler


# Función para dividir los datos en train y test
def dividir_datos(df, test_size=0.2, val_size=0.2, random_state=42):
    train_size = int(len(df) * (1 - test_size - val_size))
    val_size = int(len(df) * val_size)

    train, val, test = df.iloc[:train_size], df.iloc[train_size:train_size + val_size], df.iloc[train_size + val_size:]
    return train, val, test


# Función para entrenar el modelo VAR
def entrenar_var(train, max_lags=5):
    model = VAR(train)
    model_fitted = model.fit(max_lags)
    print(f"Número óptimo de rezagos: {model_fitted.k_ar}")
    return model_fitted


# Función para hacer predicciones y evaluar
def evaluar_modelo(modelo, test):
    lags = modelo.k_ar
    predicciones = modelo.forecast(test.values[:lags], steps=len(test))
    pred_df = pd.DataFrame(predicciones, index=test.index, columns=test.columns)

    mse = mean_squared_error(test, pred_df)
    print(f"MSE: {mse:.4f}")

    return pred_df


# Función principal
def main():
    ruta_parquet = "ruta/a/tu/dataset.parquet"
    df = cargar_datos(ruta_parquet)

    df_scaled, scaler = preprocesar_datos(df)
    train, val, test = dividir_datos(df_scaled)

    modelo_var = entrenar_var(train)
    predicciones = evaluar_modelo(modelo_var, test)

    print(predicciones.head())


if __name__ == "__main__":
    main()