import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Función para cargar datos desde un archivo Parquet
def cargar_datos(ruta_parquet):
    df = pd.read_parquet(ruta_parquet)
    return df

# Función para preprocesar los datos
def preprocesar_datos(df):
    df = df.drop(columns=['tendencia_futura_7d', 'pct_cambio_30d', 'tendencia_futura_30d'], errors='ignore')
    X = df.drop(columns=['pct_cambio_7d'])
    y = df['pct_cambio_7d']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Función para dividir los datos en train, val y test
def dividir_datos(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Función para buscar los mejores hiperparámetros (p,d,q)
def buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val):
    mejor_aic = np.inf  # AIC debe ser lo más bajo posible
    mejor_modelo = None
    mejores_parametros = None

    # Probamos diferentes combinaciones de parámetros para ARIMA
    p_values = [1, 2, 3]
    d_values = [1]
    q_values = [0, 1, 2]

    # Buscamos los mejores parámetros de ARIMA
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    modelo_arima = ARIMA(y_train, order=(p, d, q))
                    modelo_resultado = modelo_arima.fit()

                    # Obtenemos el AIC para cada modelo
                    aic = modelo_resultado.aic

                    if aic < mejor_aic:
                        mejor_aic = aic
                        mejor_modelo = modelo_resultado
                        mejores_parametros = (p, d, q)
                except:
                    continue

    print(f"Mejores hiperparámetros: p={mejores_parametros[0]}, d={mejores_parametros[1]}, q={mejores_parametros[2]}")
    return mejor_modelo


# Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.forecast(steps=len(y_test))
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Evaluación sobre test - MAE: {mae:.4f}")
    print(f"Evaluación sobre test - RMSE: {rmse:.4f}")


# Función principal
def main():
    ruta_parquet = "ruta/a/tu/dataset.parquet"
    df = cargar_datos(ruta_parquet)

    X, y, scaler = preprocesar_datos(df)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)

    # Buscamos el mejor modelo ARIMA usando el conjunto de entrenamiento y validación
    mejor_modelo = buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val)

    # Evaluamos el modelo en el conjunto de test
    evaluar_modelo(mejor_modelo, X_test, y_test)


if __name__ == "__main__":
    main()