import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV
import mlflow

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


def preprocesar_datos(df):
    df = df.drop(
        columns=["tendencia_futura_7d", "pct_cambio_30d", "tendencia_futura_30d"],
        errors="ignore",
    )
    X = df.drop(columns=["pct_cambio_7d"])
    y = df["pct_cambio_7d"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def dividir_datos_temporalmente(X, y, train_size=0.8, val_size=0.1):
    n = len(X)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val):
    parametros = {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    dt = DecisionTreeRegressor()
    grid_search = GridSearchCV(dt, parametros, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_
    mejores_parametros = grid_search.best_params_

    print("Mejores Hiperparámetros:", mejores_parametros)

    y_val_pred = mejor_modelo.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(val_mse)

    print(f"Validación MSE: {val_mse:.4f}")
    print(f"Validación RMSE: {val_rmse:.4f}")
    print(f"Validación MAE: {val_mae:.4f}")
    print(f"Validación R²: {val_r2:.4f}")

    return mejor_modelo, mejores_parametros


def evaluar_modelo(modelo, X_test, y_test):
    predictions = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    # Accuracy direccional (comparar signo de la predicción y del valor real)
    pred_sign = np.sign(predictions)
    real_sign = np.sign(y_test)
    directional_acc = accuracy_score(real_sign, pred_sign)


    print("\nMétricas del modelo:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy Direccional: {directional_acc:.4f}")

    return mse, mae, rmse, r2, directional_acc


def main():
    mlflow.set_tracking_uri(r"file:../../mlruns")
    mlflow.set_experiment("Arboles Decision")

    ruta_parquet = r"../../../../data/parquet/data_7d.parquet"
    df = cargar_datos(ruta_parquet)

    X, y, scaler = preprocesar_datos(df)

    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos_temporalmente(X, y)

    with mlflow.start_run(run_name=f"Arbol Decision 7d regresion"):
        mejor_modelo, mejores_parametros = buscar_mejores_hiperparametros(
            X_train, y_train, X_val, y_val
        )

        mse, mae, rmse, r2, directional_acc = evaluar_modelo(mejor_modelo, X_test, y_test)

        mlflow.log_params(mejores_parametros)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "directional_acc":directional_acc})

if __name__ == "__main__":
    main()