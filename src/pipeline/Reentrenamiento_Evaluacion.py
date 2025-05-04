import os
import joblib
from xgboost import XGBRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d, cargar_datos_30d
from utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_7d, preprocesar_datos_30d
from utils.modelos_sklearn.F_metricas import evaluar_modelo_7d, evaluar_modelo_30d
from utils.paths import get_path



def seleccionar_variables_mutual_info_k15(X, y, k_vars=15):
    mi = mutual_info_regression(X, y)
    mi_series = pd.Series(mi, index=X.columns)
    columnas_seleccionadas = mi_series.sort_values(ascending=False).head(k_vars).index.tolist()
    return X[columnas_seleccionadas], columnas_seleccionadas



def accuracy_direccion(y_true, y_pred):
    """Calcula la accuracy direccional."""
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction)


def evaluar_modelo(y_true, y_pred, modelo_nombre):
    """Evalúa el modelo con varias métricas."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    acc_dir = accuracy_direccion(y_true, y_pred)
    print(f"Métricas {modelo_nombre}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy Direccional: {acc_dir:.4f}")
    return mse, mae, rmse, r2, acc_dir


def entrenar_evaluar_y_reentrenar_modelo(tecnica, test_size=0.2):
    """
    Entrena, evalúa y reentrena el modelo SVR o ARIMA.

    Args:
        tecnica (str): 'svr' o 'xgboost' para especificar el modelo.
        test_size (float): Proporción de datos a usar como conjunto de prueba
                           durante la evaluación inicial.
    Returns:
        None
    """
    os.makedirs("modelos", exist_ok=True)

    if tecnica == "svr":
        # Cargar y preparar datos de entrenamiento
        parquet_file_path_train = get_path('data', 'parquet', 'data_7d.parquet')
        df_train = cargar_datos_7d(parquet_file_path_train)
        df_train_sin_nan = df_train.dropna(subset=["pct_cambio_7d"])
        y_train = df_train_sin_nan["pct_cambio_7d"]
        X_train = df_train_sin_nan.drop(columns=["pct_cambio_7d"])

        # Selección de columnas con mutual_info_regression
        X_train_sel, columnas_seleccionadas = seleccionar_variables_mutual_info_k15(X_train, y_train, k_vars=15)
        print(f"Columnas de SVR ({len(columnas_seleccionadas)}):")
        print(columnas_seleccionadas)

        # Imputación y escalado
        imputador = SimpleImputer(strategy="mean")
        X_train_sel = imputador.fit_transform(X_train_sel)
        X_train_sel = pd.DataFrame(X_train_sel, columns=columnas_seleccionadas)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)

        # Entrenar SVR con mejores parámetros
        best_params_svr = {'kernel': 'linear', 'epsilon': 0.5, 'C': 0.1}
        modelo_svr = SVR(**best_params_svr)
        modelo_svr.fit(X_train_scaled, y_train)

        # Ver resultados
        print("Reentrenamiento SVR:")
        metricas_svr_nuevos = evaluar_modelo_7d(modelo_svr, X_train_scaled, y_train)

        # ------------------------
        # Evaluación en nuevos datos
        # ------------------------
        parquet_file_path_nuevos = get_path('data', 'parquet nuevos', 'data_7d_nuevos.parquet')
        df_nuevos = cargar_datos_7d(parquet_file_path_nuevos)
        df_nuevos_sin_nan = df_nuevos.dropna(subset=["pct_cambio_7d"])
        y_nuevos = df_nuevos_sin_nan["pct_cambio_7d"]
        X_nuevos = df_nuevos_sin_nan.drop(columns=["pct_cambio_7d"])

        # Asegurar columnas consistentes
        for col in columnas_seleccionadas:
            if col not in X_nuevos.columns:
                X_nuevos[col] = np.nan
        X_nuevos_sel = X_nuevos[columnas_seleccionadas]

        # Imputar y escalar nuevos
        X_nuevos_imputado = imputador.transform(X_nuevos_sel)
        X_nuevos_escalado = scaler.transform(X_nuevos_imputado)

        print("Métricas SVR en datos nuevos:")
        metricas_svr_nuevos = evaluar_modelo_7d(modelo_svr, X_nuevos_escalado, y_nuevos)

        # ------------------------
        # Reentrenamiento con todos los datos
        # ------------------------
        df_final = pd.concat([df_train_sin_nan, df_nuevos_sin_nan])
        y_final = df_final["pct_cambio_7d"]
        X_final = df_final.drop(columns=["pct_cambio_7d"])

        # Usar las mismas columnas seleccionadas
        for col in columnas_seleccionadas:
            if col not in X_final.columns:
                X_final[col] = np.nan
        X_final_sel = X_final[columnas_seleccionadas]

        # Imputar y escalar
        X_final_imputado = imputador.fit_transform(X_final_sel)
        X_final_escalado = scaler.fit_transform(X_final_imputado)

        # Reentrenar modelo final
        modelo_svr_final = SVR(**best_params_svr)
        modelo_svr_final.fit(X_final_escalado, y_final)

        # Guardar modelo final
        ruta_modelo_final = "modelos/SVR.joblib"
        ruta_scaler_final = "modelos/SVR_scaler.joblib"
        joblib.dump(modelo_svr_final, ruta_modelo_final)
        joblib.dump(scaler, ruta_scaler_final)

        print(f"Modelo SVR final guardado en: {ruta_modelo_final}")
        print(f"Scaler final guardado en: {ruta_scaler_final}")

        # Evaluación final
        pred_final = modelo_svr_final.predict(X_final_escalado)
        evaluar_modelo(y_final, pred_final, "SVR Final")


    elif tecnica == "xgboost":

        print(f"\n--- Entrenando, evaluando y reentrenando modelo XGBoost ---")

        """

        joblib.dump(modelo_xgb_final, "modelos/XGB.joblib")
        joblib.dump(scaler_final, "modelos/XGB_scaler.joblib")
        
        pred_final_total = modelo_xgb_final.predict(X_scaled_final)
        evaluar_modelo(y_final, pred_final_total, "XGBoost (Final)")

        """
if __name__ == "__main__":
    # Entrenar, evaluar y reentrenar SVR
    entrenar_evaluar_y_reentrenar_modelo(tecnica="svr", test_size=0.2)

    # Entrenar, evaluar y reentrenar ARIMA
    entrenar_evaluar_y_reentrenar_modelo(tecnica="xgboost", test_size=0.2)