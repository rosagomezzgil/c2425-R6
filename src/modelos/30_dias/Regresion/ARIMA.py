"""
    Modelo de REGRESIÓN.
    Buen modelo para predecir tendencias en series temporales. Como mejor funciona es con datos estacionarios,
    es decir, aquellos cuya distribución no cambia con el tiempo.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# 1. Cargar los datos
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])  # Asegurar que la fecha esté en formato datetime
    df.set_index('date', inplace=True)
    return df


# 2. Preprocesar los datos
def preprocess_data(df, horizonte):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['close_binance']])

    x, y = [], []
    for i in range(len(df_scaled) - horizonte):
        x.append(df_scaled[i:i + horizonte])
        y.append(df_scaled[i + horizonte])

    x, y = np.array(x), np.array(y)
    return x, y, scaler





# 5. Ejecutar el flujo
def main():
    csv_file = 'merged_data.csv'  # Asegúrate de tener este archivo
    horizonte = 7  # 7 días a predecir

    df = load_data(csv_file)
    x, y, scaler = preprocess_data(df, horizonte)

    #Split en Train (70%) y Temp (30%)
    X_train, X_temp, y_train_7, y_temp_7 = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    #Split de Temp en Validación (15%) y Test (15%)
    X_val, X_test, y_val_7, y_test_7 = train_test_split(
        X_temp, y_temp_7, test_size=0.5, random_state=42
    )

    #hacemos el modelo
    #Configurar MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Bitcoin_Percentage_Change_ARIMA")

    #Entrenar modelo ARIMA
    with mlflow.start_run():
        print("📈 Entrenando modelo AutoARIMA...")
        model = pm.auto_arima(y_train_7, seasonal=False, stepwise=True, suppress_warnings=True, trace=True)

        #Hacer predicciones
        y_pred = model.predict(n_periods=len(y_val_7))

        #Evaluación del modelo
        mae = mean_absolute_error(y_val_7, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val_7, y_pred))
        r2 = r2_score(y_val_7, y_pred)

        #Registrar en MLflow
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, "arima_regression_model")

        print(f"✔ Evaluación del modelo:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")



if __name__ == "__main__":
    main()


