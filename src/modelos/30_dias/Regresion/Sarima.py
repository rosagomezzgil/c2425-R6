import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


# Función para cargar datos desde un archivo Parquet
def cargar_datos(ruta_parquet):
    df = pd.read_parquet(ruta_parquet)
    return df


# Función para preprocesar los datos
def preprocesar_datos(df, variable_objetivo='pct_cambio_30d'):
    df = df[[variable_objetivo]].dropna()  # Solo mantenemos la variable objetivo y eliminamos valores nulos
    return df


# Función para dividir los datos en train y test
def dividir_datos(df, test_size=0.2, val_size=0.2):
    train_size = int(len(df) * (1 - test_size - val_size))
    val_size = int(len(df) * val_size)

    train, val, test = df.iloc[:train_size], df.iloc[train_size:train_size + val_size], df.iloc[train_size + val_size:]
    return train, val, test


# Función para entrenar el modelo SARIMA
def entrenar_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fitted = model.fit(disp=False)
    print(model_fitted.summary())
    return model_fitted


# Función para hacer predicciones y evaluar
def evaluar_modelo(modelo, test):
    predicciones = modelo.forecast(steps=len(test))
    mse = mean_squared_error(test, predicciones)
    print(f"MSE: {mse:.4f}")

    return predicciones


# Función principal
def main():
    ruta_parquet = "ruta/a/tu/dataset.parquet"
    df = cargar_datos(ruta_parquet)

    df_preprocesado = preprocesar_datos(df)
    train, val, test = dividir_datos(df_preprocesado)

    modelo_sarima = entrenar_sarima(train)
    predicciones = evaluar_modelo(modelo_sarima, test)

    print(predicciones.head())


if __name__ == "__main__":
    main()