import pandas as pd
import numpy as np
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Función para cargar datos desde un archivo Parquet
def cargar_datos(ruta_parquet):
    df = pd.read_parquet(ruta_parquet)
    return df


# Función para preprocesar los datos
def preprocesar_datos(df, variable_objetivo="pct_cambio_30d"):
    df = df[[variable_objetivo]].dropna()  # Solo mantenemos la variable objetivo y eliminamos valores nulos
    return df


# Función para dividir los datos en train, validación y test
def dividir_datos(df, test_size=0.1, val_size=0.1):
    train_size = int(len(df) * (1 - test_size - val_size))
    val_size = int(len(df) * val_size)

    train, val, test = (
        df.iloc[:train_size],
        df.iloc[train_size : train_size + val_size],
        df.iloc[train_size + val_size :],
    )
    return train, val, test


# Función para buscar los mejores hiperparámetros
def optimizar_sarima(train, val, p_vals=[0, 1, 2], d_vals=[0, 1], q_vals=[0, 1, 2],
                      P_vals=[0, 1], D_vals=[0, 1], Q_vals=[0, 1], s=30):
    mejor_mse = float("inf")
    mejor_modelo = None
    mejor_params = None

    print("\n🔍 Buscando los mejores hiperparámetros SARIMA...")

    for (p, d, q, P, D, Q) in product(p_vals, d_vals, q_vals, P_vals, D_vals, Q_vals):
        try:
            order = (p, d, q)
            seasonal_order = (P, D, Q, s)

            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fitted = model.fit(disp=False)

            val_pred = model_fitted.forecast(steps=len(val))
            mse = mean_squared_error(val, val_pred)

            print(f"Probando SARIMA{order} x {seasonal_order} - MSE Validación: {mse:.4f}")

            if mse < mejor_mse:
                mejor_mse = mse
                mejor_modelo = model_fitted
                mejor_params = (order, seasonal_order)

        except Exception as e:
            print(f"⚠️ Error con SARIMA{order} x {seasonal_order}: {e}")

    print(f"\n✅ Mejor modelo: SARIMA{mejor_params[0]} x {mejor_params[1]} con MSE {mejor_mse:.4f}")
    return mejor_modelo


# Función para evaluar el modelo
def evaluar_modelo(modelo, test):
    predicciones = modelo.forecast(steps=len(test))

    # Calcular métricas
    mse = mean_squared_error(test, predicciones)
    mae = mean_absolute_error(test, predicciones)
    rmse = np.sqrt(mse)
    r2 = r2_score(test, predicciones)

    print(f"\n📊 Evaluación del modelo SARIMA:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return predicciones


# Función principal
def main():
    ruta_parquet = r"../../../../data/parquet/data_30d.parquet"
    df = cargar_datos(ruta_parquet)

    df_preprocesado = preprocesar_datos(df)
    train, val, test = dividir_datos(df_preprocesado)

    modelo_sarima = optimizar_sarima(train, val)
    predicciones = evaluar_modelo(modelo_sarima, test)

    print("Primeras predicciones:")
    print(predicciones.head())


if __name__ == "__main__":
    main()
