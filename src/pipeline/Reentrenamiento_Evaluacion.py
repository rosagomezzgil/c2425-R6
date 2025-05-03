import os
import joblib
from xgboost import XGBRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from src.utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d, cargar_datos_30d
from src.utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_7d, preprocesar_datos_30d
from src.utils.modelos_sklearn.F_metricas import evaluar_modelo_7d, evaluar_modelo_30d
from src.utils.paths import get_path


def seleccionar_variables_f_regression_k10(X, y, k_vars=15):
    """
    Selecciona las mejores k variables usando f_regression con un valor fijo de k.
    """
    original_columns = X.columns
    selector_f = SelectKBest(score_func=mutual_info, k=min(k_vars, X.shape[1]))
    X_kbest_f = selector_f.fit_transform(X, y)
    selected_columns_indices_f = selector_f.get_support()
    selected_columns_f = original_columns[selected_columns_indices_f]
    return pd.DataFrame(X_kbest_f, columns=selected_columns_f, index=X.index)


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
        tecnica (str): 'svr' o 'arima' para especificar el modelo.
        test_size (float): Proporción de datos a usar como conjunto de prueba
                           durante la evaluación inicial.
    Returns:
        None
    """
    os.makedirs("modelos", exist_ok=True)

    if tecnica == "svr":
        print(f"\n--- Entrenando, evaluando y reentrenando modelo SVR ---")

        # Paso 1: Cargar y preparar los datos iniciales
        parquet_file_path_7d = get_path('data', 'parquet', 'data_7d.parquet')
        df_7d = cargar_datos_7d(parquet_file_path_7d)
        df_7d_sin_nan = df_7d.dropna(subset=["pct_cambio_7d"])
        y_7d = df_7d_sin_nan["pct_cambio_7d"]
        X_original_7d = df_7d_sin_nan.drop(columns=["pct_cambio_7d"])

        # Dividir ANTES de cualquier selección o escalado
        X_train, X_test_eval, y_train, y_test_eval = train_test_split(
            X_original_7d, y_7d, test_size=test_size, random_state=42
        )

        # Paso 2: Selección de variables SOLO con datos de entrenamiento
        X_train_sel = seleccionar_variables_f_regression_k10(X_train, y_train)

        # Aplicar la misma selección al conjunto de prueba
        X_test_sel = X_test_eval[X_train_sel.columns]

        # Paso 3: Escalado con datos de entrenamiento
        scaler_eval = StandardScaler()
        X_scaled_train = scaler_eval.fit_transform(X_train_sel)
        X_scaled_test_eval = scaler_eval.transform(X_test_sel)

        # Entrenar y evaluar
        best_params_svr = {'kernel': 'linear', 'epsilon': 0.5, 'C': 0.1}
        modelo_svr_eval = SVR(**best_params_svr)
        modelo_svr_eval.fit(X_scaled_train, y_train)
        metricas_svr_eval = evaluar_modelo_7d(modelo_svr_eval, X_scaled_test_eval, y_test_eval)
        print("Métricas SVR en el conjunto de prueba inicial:", metricas_svr_eval)

        # Guardar modelo y escalador
        ruta_modelo_svr_eval = "modelos/SVR_eval.joblib"
        ruta_scaler_svr_eval = "modelos/SVR_scaler_eval.joblib"
        joblib.dump(modelo_svr_eval, ruta_modelo_svr_eval)
        joblib.dump(scaler_eval, ruta_scaler_svr_eval)
        print(f"Modelo SVR (evaluación inicial) guardado en: {ruta_modelo_svr_eval}")
        print(f"Escalador SVR (evaluación inicial) guardado en: {ruta_scaler_svr_eval}")

        # Paso 4: Obtener y preparar los nuevos datos (NUNCA usados antes)
        parquet_file_path_nuevos_7d = get_path('data', 'parquet nuevos', 'data_7d_nuevos.parquet')
        df_nuevos_7d = cargar_datos_7d(parquet_file_path_nuevos_7d)
        df_nuevos_7d_sin_nan = df_nuevos_7d.dropna(subset=["pct_cambio_7d"])
        y_nuevos_7d = df_nuevos_7d_sin_nan["pct_cambio_7d"]
        X_original_nuevos_7d = df_nuevos_7d_sin_nan.drop(columns=["pct_cambio_7d"])

        # Selección y escalado coherente con el modelo
        if set(X_train_sel.columns).issubset(set(X_original_nuevos_7d.columns)):
            X_seleccionado_nuevos_7d = X_original_nuevos_7d[X_train_sel.columns]
            X_scaled_nuevos_7d = scaler_eval.transform(X_seleccionado_nuevos_7d)
            metricas_svr_nuevos = evaluar_modelo_7d(modelo_svr_eval, X_scaled_nuevos_7d, y_nuevos_7d)
            print("Métricas SVR en datos nuevos:", metricas_svr_nuevos)
        else:
            print(
                "Advertencia: Las columnas de los datos nuevos no coinciden con las del entrenamiento inicial para SVR.")
            metricas_svr_nuevos = None

        # Paso 5: Reentrenar el modelo usando todos los datos (entrenamiento + validación + test + nuevos)
        df_final_7d = pd.concat([df_7d_sin_nan, df_nuevos_7d_sin_nan])
        y_final_7d = df_final_7d["pct_cambio_7d"]
        X_original_final_7d = df_final_7d.drop(columns=["pct_cambio_7d"])

        # Dividir los datos en entrenamiento y prueba para el reentrenamiento final
        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
            X_original_final_7d, y_final_7d, test_size=0.2, random_state=42
        )

        # Selección de características solo sobre los datos de entrenamiento final
        X_train_sel_final = seleccionar_variables_f_regression_k10(X_train_final, y_train_final, k_vars=10)

        # Escalado solo sobre los datos de entrenamiento final
        scaler_final_svr = StandardScaler()
        X_scaled_train_final = scaler_final_svr.fit_transform(X_train_sel_final)

        # Transformar los datos de test con el escalador entrenado
        X_scaled_test_final = scaler_final_svr.transform(X_test_final[X_train_sel_final.columns])

        # Entrenar el modelo con los datos completos escalados
        modelo_svr_final = SVR(**best_params_svr)
        modelo_svr_final.fit(X_scaled_train_final, y_train_final)

        # Evaluar el modelo en los datos de prueba
        pred_svr_final = modelo_svr_final.predict(X_scaled_test_final)
        evaluar_modelo(y_test_final, pred_svr_final, "SVR (Reentrenamiento Final)")

        # Reentrenar el modelo con todos los datos (entrenamiento + validación + test + nuevos datos)
        X_final = pd.concat([X_train_final, X_test_final])
        y_final = pd.concat([y_train_final, y_test_final])

        # Selección de características para los datos completos
        X_final_sel = seleccionar_variables_f_regression_k10(X_final, y_final, k_vars=10)

        # Escalado de todos los datos con el escalador entrenado
        X_scaled_final = scaler_final_svr.fit_transform(X_final_sel)

        # Reentrenar el modelo con todos los datos disponibles
        modelo_svr_final.fit(X_scaled_final, y_final)

        # Guardar el modelo y el escalador final
        ruta_modelo_svr_final = "modelos/SVR_final.joblib"
        ruta_scaler_svr_final = "modelos/SVR_scaler_final.joblib"
        joblib.dump(modelo_svr_final, ruta_modelo_svr_final)
        joblib.dump(scaler_final_svr, ruta_scaler_svr_final)
        print(f"Modelo SVR final reentrenado y guardado en: {ruta_modelo_svr_final}")
        print(f"Escalador SVR final guardado en: {ruta_scaler_svr_final}")

        # Evaluación final
        pred_svr_final_total = modelo_svr_final.predict(X_scaled_final)
        evaluar_modelo(y_final, pred_svr_final_total, "SVR (Final)")

    elif tecnica == "xgboost":

        print(f"\n--- Entrenando, evaluando y reentrenando modelo XGBoost ---")
        parquet_file_path_30d = get_path('data', 'parquet', 'data_30d.parquet')
        df_30d = cargar_datos_30d(parquet_file_path_30d)
        df_30d_sin_nan = df_30d.dropna(subset=["pct_cambio_30d"])
        y_30d = df_30d_sin_nan["pct_cambio_30d"]
        X_original_30d = df_30d_sin_nan.drop(columns=["pct_cambio_30d"])

        # Dividir los datos
        X_train, X_test_eval, y_train, y_test_eval = train_test_split(
            X_original_30d, y_30d, test_size=test_size, random_state=42
        )

        # Selección de características
        X_train_sel = seleccionar_variables_f_regression_k10(X_train, y_train, k_vars=15)
        X_test_sel = X_test_eval[X_train_sel.columns]

        # Escalado
        scaler_eval = StandardScaler()
        X_scaled_train = scaler_eval.fit_transform(X_train_sel)
        X_scaled_test = scaler_eval.transform(X_test_sel)

        # Entrenar y evaluar modelo
        best_params_xgb = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}
        modelo_xgb_eval = XGBRegressor(**best_params_xgb)
        modelo_xgb_eval.fit(X_scaled_train, y_train)

        # Evaluación
        pred_xgb_eval = modelo_xgb_eval.predict(X_scaled_test)
        evaluar_modelo(y_test_eval, pred_xgb_eval, "XGBoost (Evaluación Inicial)")

        # Guardar modelo y escalador
        joblib.dump(modelo_xgb_eval, "modelos/XGB_eval.joblib")
        joblib.dump(scaler_eval, "modelos/XGB_scaler_eval.joblib")

        # Cargar nuevos datos
        parquet_file_path_nuevos_30d = get_path('data', 'parquet nuevos', 'data_30d_nuevos.parquet')
        df_nuevos_30d = cargar_datos_30d(parquet_file_path_nuevos_30d)
        df_nuevos_30d_sin_nan = df_nuevos_30d.dropna(subset=["pct_cambio_30d"])
        y_nuevos_30d = df_nuevos_30d_sin_nan["pct_cambio_30d"]
        X_nuevos_30d = df_nuevos_30d_sin_nan.drop(columns=["pct_cambio_30d"])

        if set(X_train_sel.columns).issubset(X_nuevos_30d.columns):
            X_nuevos_sel = X_nuevos_30d[X_train_sel.columns]
            X_nuevos_scaled = scaler_eval.transform(X_nuevos_sel)
            pred_xgb_nuevos = modelo_xgb_eval.predict(X_nuevos_scaled)
            evaluar_modelo(y_nuevos_30d, pred_xgb_nuevos, "XGBoost (Datos Nuevos)")

        else:
            print("Advertencia: Las columnas de los datos nuevos no coinciden con las del entrenamiento.")

        # Reentrenamiento con todos los datos
        df_final_30d = pd.concat([df_30d_sin_nan, df_nuevos_30d_sin_nan])
        y_final = df_final_30d["pct_cambio_30d"]
        X_final = df_final_30d.drop(columns=["pct_cambio_30d"])
        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
            X_final, y_final, test_size=0.2, random_state=42
        )
        X_train_sel_final = seleccionar_variables_f_regression_k10(X_train_final, y_train_final, k_vars=15)
        X_test_sel_final = X_test_final[X_train_sel_final.columns]

        scaler_final = StandardScaler()
        X_scaled_train_final = scaler_final.fit_transform(X_train_sel_final)
        X_scaled_test_final = scaler_final.transform(X_test_sel_final)

        modelo_xgb_final = XGBRegressor(**best_params_xgb)
        modelo_xgb_final.fit(X_scaled_train_final, y_train_final)

        pred_xgb_final = modelo_xgb_final.predict(X_scaled_test_final)
        evaluar_modelo(y_test_final, pred_xgb_final, "XGBoost (Reentrenamiento Final)")

        # Entrenar con todos los datos disponibles

        X_final_sel = seleccionar_variables_f_regression_k10(X_final, y_final, k_vars=15)
        X_scaled_final = scaler_final.fit_transform(X_final_sel)
        modelo_xgb_final.fit(X_scaled_final, y_final)

        joblib.dump(modelo_xgb_final, "modelos/XGB_final.joblib")
        joblib.dump(scaler_final, "modelos/XGB_scaler_final.joblib")

        pred_final_total = modelo_xgb_final.predict(X_scaled_final)
        evaluar_modelo(y_final, pred_final_total, "XGBoost (Final)")


if __name__ == "__main__":
    # Entrenar, evaluar y reentrenar SVR
    entrenar_evaluar_y_reentrenar_modelo(tecnica="svr", test_size=0.2)

    # Entrenar, evaluar y reentrenar ARIMA
    entrenar_evaluar_y_reentrenar_modelo(tecnica="xgboost", test_size=0.2)