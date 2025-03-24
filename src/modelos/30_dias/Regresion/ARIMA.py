"""
    Modelo de REGRESIÓN.
    Buen modelo para predecir tendencias en series temporales. Como mejor funciona es con datos estacionarios,
    es decir, aquellos cuya distribución no cambia con el tiempo.
"""

import pandas as pd
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Función para cargar datos desde un archivo Parquet
def cargar_datos(ruta_parquet):
    df = pd.read_parquet(ruta_parquet)
    return df


# Función para preprocesar los datos
def preprocesar_datos(df):
    df = df.drop(columns=['tendencia_futura_7d', 'pct_cambio_30d', 'tendencia_futura_30d'], errors='ignore')
    return df['close']


# Función para dividir los datos en entrenamiento y prueba
def dividir_datos(serie, test_size=0.2, random_state=42):
    train_size = int(len(serie) * (1 - test_size))
    train, test = serie[:train_size], serie[train_size:]
    return train, test


# Función para buscar los mejores hiperparámetros (p,d,q)
def buscar_mejores_hiperparametros(serie):
    p = d = q = range(0, 3)
    pdq_combinations = list(itertools.product(p, d, q))

    mejor_pdq = None
    mejor_aic = float("inf")

    for param in pdq_combinations:
        try:
            model = ARIMA(serie, order=param)
            result = model.fit()
            if result.aic < mejor_aic:
                mejor_aic = result.aic
                mejor_pdq = param
        except:
            continue

    print("Mejores parámetros ARIMA:", mejor_pdq)
    return mejor_pdq


# Función para entrenar y evaluar el modelo ARIMA
def entrenar_y_evaluar(serie):
    mejor_pdq = buscar_mejores_hiperparametros(serie)
    model = ARIMA(serie, order=mejor_pdq)
    result = model.fit()

    # Predicciones para los próximos 30 días
    predicciones = result.forecast(steps=30)
    real = serie[-30:]

    # Evaluación del modelo
    mae = mean_absolute_error(real, predicciones)
    rmse = mean_squared_error(real, predicciones, squared=False)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


# Función principal
def main():
    ruta_parquet = "ruta/a/tu/dataset.parquet"
    df = cargar_datos(ruta_parquet)

    # Preprocesamos y dividimos los datos
    serie = preprocesar_datos(df)
    train, test = dividir_datos(serie)

    # Entrenamos y evaluamos el modelo ARIMA
    entrenar_y_evaluar(train)


if __name__ == "__main__":
    main()


