import pandas as pd
import numpy as np
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow

# Función para cargar datos
def cargar_datos(parquet_file: str):
    """
    Carga los datos de un fichero parquet a un DataFrame de pandas

    :param parquet_file: URL del fichero parquet

    :return: pd.DataFrame
    """
    df = pd.read_parquet(parquet_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Asegurar que la fecha esté en formato datetime
    df.set_index('timestamp', inplace=True)
    return df


# Función para preprocesar los datos
def preprocesar_datos(df, variable_objetivo="pct_cambio_30d"):
    df = df[[variable_objetivo]].dropna()
    return df


# Función para dividir los datos
def dividir_datos(df, test_size=0.1, val_size=0.1):
    train_size = int(len(df) * (1 - test_size - val_size))
    val_size = int(len(df) * val_size)

    train, val, test = (
        df.iloc[:train_size],
        df.iloc[train_size : train_size + val_size],
        df.iloc[train_size + val_size :],
    )
    return train, val, test


# Validación Walk-Forward
def validar_walk_forward(train, val, order, seasonal_order):
    total_mse = []

    for i in range(len(val)):  # Recorremos la ventana deslizante
        train_fold = train.iloc[: len(train) + i]
        val_point = val.iloc[i : i + 1]  # Solo un punto de validación

        try:
            model = SARIMAX(
                train_fold,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            pred = model.forecast(steps=1)
            mse = mean_squared_error(val_point, pred)
            total_mse.append(mse)
        except Exception as e:
            print(f"⚠️ Error en validación walk-forward: {e}")
            return float("inf")  # Penalización si falla

    return np.mean(total_mse)


# Función para buscar los mejores hiperparámetros
def optimizar_sarima(train, val, p_vals=[0, 1], d_vals=[0, 1], q_vals=[0, 1],
                      P_vals=[0, 1], D_vals=[0, 1], Q_vals=[0, 1], s=30):
    mejor_mse = float("inf")
    mejor_modelo = None
    mejor_params = None

    print("\n🔍 Buscando los mejores hiperparámetros SARIMA...")

    for (p, d, q, P, D, Q) in product(p_vals, d_vals, q_vals, P_vals, D_vals, Q_vals):
        try:
            order = (p, d, q)
            seasonal_order = (P, D, Q, s)

            mse = validar_walk_forward(train, val, order, seasonal_order)
            print(f"Probando SARIMA{order} x {seasonal_order} - MSE Validación: {mse:.4f}")

            if mse < mejor_mse:
                mejor_mse = mse
                mejor_modelo = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                mejor_params = (order, seasonal_order)

        except Exception as e:
            print(f"⚠️ Error con SARIMA{order} x {seasonal_order}: {e}")

    print(f"\n✅ Mejor modelo: SARIMA{mejor_params[0]} x {mejor_params[1]} con MSE {mejor_mse:.4f}")
    return mejor_modelo, {'order':order, 'seasonal_order':seasonal_order}


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

    return mse, mae, rmse, r2


# Función principal
def main():
    mlflow.set_tracking_uri(r"file:../../mlruns")
    mlflow.set_experiment("Sarima 30d")

    ruta_parquet = r"../../../../data/parquet/data_30d.parquet"
    df = cargar_datos(ruta_parquet)

    df_preprocesado = preprocesar_datos(df)
    train, val, test = dividir_datos(df_preprocesado)

    with mlflow.start_run(run_name=f"Sarima walk forward"):
        modelo_sarima, best_params = optimizar_sarima(train, val)
        mse, mae, rmse, r2 = evaluar_modelo(modelo_sarima, test)

        mlflow.log_params(best_params)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2})


if __name__ == "__main__":
    main()
