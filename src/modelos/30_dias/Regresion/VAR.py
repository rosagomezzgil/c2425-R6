import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# Función para cargar datos desde un archivo Parquet
def cargar_datos(ruta_parquet):
    df = pd.read_parquet(ruta_parquet)
    return df


# Función para preprocesar los datos
def preprocesar_datos(df):
    df = df.drop(columns=['tendencia_futura_7d', 'pct_cambio_7d', 'tendencia_futura_30d'], errors='ignore')
    X = df.drop(columns=['pct_cambio_30d'])
    y = df[['pct_cambio_30d']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
    df_scaled = pd.concat([df_scaled, y], axis=1)  # Agregar variables objetivo sin escalar

    return df_scaled, scaler


# Función para dividir los datos en train, validación y test
def dividir_datos(df, test_size=0.2, val_size=0.2):
    train_size = int(len(df) * (1 - test_size - val_size))
    val_size = int(len(df) * val_size)

    train, val, test = df.iloc[:train_size], df.iloc[train_size:train_size + val_size], df.iloc[train_size + val_size:]
    return train, val, test


# Función para entrenar y validar el modelo VAR
def entrenar_y_validar_var(train, val, max_lags=5):
    model = VAR(train)
    best_model = None
    best_mse = float('inf')
    best_lag = None

    # Probar diferentes valores de rezagos
    for lags in range(1, max_lags + 1):
        try:
            model_fitted = model.fit(lags)
            val_pred = model_fitted.forecast(train.values[-lags:], steps=len(val))
            val_pred_df = pd.DataFrame(val_pred, index=val.index, columns=val.columns)
            mse = mean_squared_error(val, val_pred_df)

            print(f"Lags: {lags}, MSE de validación: {mse:.4f}")

            if mse < best_mse:
                best_mse = mse
                best_model = model_fitted
                best_lag = lags
        except Exception as e:
            print(f"Error con {lags} rezagos: {e}")

    print(f"Mejor número de rezagos: {best_lag} con MSE {best_mse:.4f}")
    return best_model


# Función para evaluar el modelo
def evaluar_modelo(modelo, test):
    if modelo is None:
        print("El modelo no fue entrenado correctamente.")
        return None

    lags = modelo.k_ar
    predicciones = modelo.forecast(test.values[:lags], steps=len(test))
    pred_df = pd.DataFrame(predicciones, index=test.index, columns=test.columns)

    mse = mean_squared_error(test, pred_df)
    print(f"MSE en test: {mse:.4f}")

    return pred_df


# Función principal
def main():
    ruta_parquet = r"../../../../data/parquet/data_7d.parquet"
    df = cargar_datos(ruta_parquet)
    df_scaled, scaler = preprocesar_datos(df)
    train, val, test = dividir_datos(df_scaled)

    modelo_var = entrenar_y_validar_var(train, val)
    predicciones = evaluar_modelo(modelo_var, test)

    if predicciones is not None:
        print(predicciones.head())


if __name__ == "__main__":
    main()
