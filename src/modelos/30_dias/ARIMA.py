import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow

# Función para cargar datos desde un archivo Parquet
def cargar_datos(parquet_file: str):
    df = pd.read_parquet(parquet_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


# Función para preprocesar los datos
def preprocesar_datos(df):
    df = df.drop(columns=["tendencia_futura_7d", "pct_cambio_7d", "tendencia_futura_30d"], errors="ignore")
    X = df.drop(columns=["pct_cambio_30d"])
    y = df["pct_cambio_30d"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# Función para dividir los datos
def dividir_datos(X, y, test_size=0.1, val_size=0.1):
    n = len(X)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))

    return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]


# Función para buscar los mejores hiperparámetros (p,d,q)
def buscar_mejores_hiperparametros(y_train, y_val):
    mejor_rmse = np.inf
    mejor_modelo = None
    mejores_parametros = None

    p_values = [1, 2, 3]
    d_values = [1]
    q_values = [0, 1, 2]

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    modelo_arima = ARIMA(y_train, order=(p, d, q))
                    modelo_resultado = modelo_arima.fit()

                    # Validamos en y_val
                    y_pred_val = modelo_resultado.forecast(steps=len(y_val))
                    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

                    print(f"(p={p}, d={d}, q={q}) → RMSE validación: {rmse_val:.4f}")

                    if rmse_val < mejor_rmse:
                        mejor_rmse = rmse_val
                        mejor_modelo = modelo_resultado
                        mejores_parametros = (p, d, q)

                except Exception as e:
                    print(f"Error con (p={p}, d={d}, q={q}): {e}")

    print(
        f"\n Mejor modelo según validación: p={mejores_parametros[0]}, d={mejores_parametros[1]}, q={mejores_parametros[2]}")
    return mejor_modelo, dict(zip(['p', 'd', 'q'], mejores_parametros))


# Función para evaluar el modelo
def evaluar_modelo(modelo, y_test):
    y_pred = modelo.forecast(steps=len(y_test))

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluación sobre test - MAE: {mae:.4f}")
    print(f"Evaluación sobre test - MSE: {mse:.4f}")
    print(f"Evaluación sobre test - RMSE: {rmse:.4f}")
    print(f"Evaluación sobre test - R²: {r2:.4f}")

    return mae, mse, rmse, r2


# Función principal
def main():
    mlflow.set_tracking_uri(r"file:../../mlruns")
    mlflow.set_experiment("ARIMA 30d")

    ruta_parquet = r"../../../../data/parquet/data_30d.parquet"
    df = cargar_datos(ruta_parquet)

    X, y, scaler = preprocesar_datos(df)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)

    with mlflow.start_run(run_name=f"ARIMA"):
        mejor_modelo, best_params = buscar_mejores_hiperparametros(y_train, y_val)

        mae, mse, rmse, r2 = evaluar_modelo(mejor_modelo, y_test)

        mlflow.log_params(best_params)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2})

if __name__ == "__main__":
    main()