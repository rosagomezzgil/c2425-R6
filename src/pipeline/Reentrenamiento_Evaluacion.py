import os
import joblib
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

# Importar funciones específicas de tu proyecto
from src.utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d, cargar_datos_30d
from src.utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_7d, preprocesar_datos_30d
from src.utils.modelos_sklearn.F_metricas import evaluar_modelo_7d, evaluar_modelo_30d
from src.utils.paths import get_path


def seleccionar_variables_f_regression_k10(X, y, k_vars=10):
    """
    Selecciona las mejores k variables usando f_regression con un valor fijo de k.
    """
    original_columns = X.columns
    selector_f = SelectKBest(score_func=f_regression, k=min(k_vars, X.shape[1]))
    X_kbest_f = selector_f.fit_transform(X, y)
    selected_columns_indices_f = selector_f.get_support()
    selected_columns_f = original_columns[selected_columns_indices_f]
    return pd.DataFrame(X_kbest_f, columns=selected_columns_f, index=X.index)


def reentrenar_y_evaluar_modelo(tecnica, reentrenar_final=False):
    """
    Reentrena y evalúa el modelo SVR o ARIMA, con la opción de reentrenar
    el modelo final con todos los datos disponibles.

    Args:
        tecnica (str): 'svr' o 'arima' para especificar el modelo.
        reentrenar_final (bool): Si es True, reentrena el modelo con todos los
            datos después de la evaluación inicial.
    Returns:
        None
    """
    if tecnica == "svr":
        print("\nReentrenando y evaluando modelo SVR...")
        # Cargar todos los datos disponibles para 7 días
        parquet_file_path_7d = get_path('data', 'parquet', 'data_7d.parquet') # Para entrenar
        df_entrenamiento_7d = cargar_datos_7d(parquet_file_path_7d)
        y_entrenamiento_7d = df_entrenamiento_7d["pct_cambio_7d"]  # Separar la variable objetivo
        X_original_entrenamiento_7d = df_entrenamiento_7d.drop(columns=["pct_cambio_7d"])

        # Selección de variables: f_regression con k=10
        X_seleccionado_entrenamiento_7d = seleccionar_variables_f_regression_k10(X_original_entrenamiento_7d, y_entrenamiento_7d, k_vars=10)

        # Escalar las características seleccionadas
        scaler = StandardScaler()
        X_scaled_entrenamiento_7d = scaler.fit_transform(X_seleccionado_entrenamiento_7d)
        X_scaled_df_entrenamiento_7d = pd.DataFrame(X_scaled_entrenamiento_7d, columns=X_seleccionado_entrenamiento_7d.columns, index=X_seleccionado_entrenamiento_7d.index)

        # Definir y entrenar el modelo SVR con hiperparámetros fijos
        best_params_svr = {'kernel': 'linear', 'epsilon': 0.1, 'C': 0.1}
        modelo_svr = SVR(**best_params_svr)
        modelo_svr.fit(X_scaled_df_entrenamiento_7d, y_entrenamiento_7d)  # Usar los datos escalados

        # Guardar el modelo y el escalador
        os.makedirs("modelos", exist_ok=True)
        ruta_modelo_svr = "modelos/SVR.joblib"
        ruta_scaler = "modelos/SVR.joblib"
        joblib.dump(modelo_svr, ruta_modelo_svr)
        joblib.dump(scaler, ruta_scaler)  # Guardar el escalador
        print(f"Modelo SVR reentrenado y guardado en: {ruta_modelo_svr}")
        print(f"Escalador guardado en: {ruta_scaler}")

        # Cargar y evaluar con datos nuevos
        parquet_file_path_nuevos_7d = get_path('data', 'nuevos', 'data_7d.parquet') # Para evaluar
        df_nuevos_7d = cargar_datos_7d(parquet_file_path_nuevos_7d)
        y_nuevos_7d = df_nuevos_7d["pct_cambio_7d"]
        X_original_nuevos_7d = df_nuevos_7d.drop(columns=["pct_cambio_7d"])

        X_seleccionado_nuevos_7d = X_original_nuevos_7d[X_seleccionado_entrenamiento_7d.columns] # Seleccionar las mismas columnas
        X_scaled_nuevos_7d = scaler.transform(X_seleccionado_nuevos_7d)
        X_scaled_df_nuevos_7d = pd.DataFrame(X_scaled_nuevos_7d, columns=X_seleccionado_nuevos_7d.columns, index=X_seleccionado_nuevos_7d.index)
        metricas_svr = evaluar_modelo_7d(modelo_svr, X_scaled_df_nuevos_7d, y_nuevos_7d)
        print("Métricas SVR en datos nuevos:", metricas_svr)

        if reentrenar_final:
            print("\nReentrenando el modelo SVR final con todos los datos...")
            # Volver a cargar todos los datos para el reentrenamiento final
            df_final_7d = pd.concat([df_entrenamiento_7d, df_nuevos_7d]) # Concatenamos los dataframes
            y_final_7d = df_final_7d["pct_cambio_7d"]
            X_original_final_7d = df_final_7d.drop(columns=["pct_cambio_7d"])
             # Selección de variables: f_regression con k=10
            X_seleccionado_final_7d = seleccionar_variables_f_regression_k10(X_original_final_7d, y_final_7d, k_vars=10)
            scaler_final = StandardScaler() # Nuevo Scaler
            X_scaled_final_7d = scaler_final.fit_transform(X_seleccionado_final_7d)  # ¡Reajustar el escalador!
            X_scaled_df_final_7d = pd.DataFrame(X_scaled_final_7d, columns=X_seleccionado_final_7d.columns, index=X_seleccionado_final_7d.index)
            modelo_svr_final = SVR(**best_params_svr)
            modelo_svr_final.fit(X_scaled_df_final_7d, y_final_7d)
            joblib.dump(modelo_svr_final, ruta_modelo_svr.replace(".joblib", "_final.joblib"))
            joblib.dump(scaler_final, ruta_scaler.replace(".joblib", "_final.joblib")) # Guardar nuevo scaler
            print("Modelo SVR final reentrenado y guardado.")

    elif tecnica == "arima":
        print("\nReentrenando y evaluando modelo ARIMA...")
        # Cargar todos los datos disponibles para 30 días
        parquet_file_path_30d = get_path('data', 'parquet', 'data_30d.parquet') # Para entrenar
        df_entrenamiento_30d = cargar_datos_30d(parquet_file_path_30d)
        y_entrenamiento_30d = df_entrenamiento_30d["pct_cambio_30d"]

        # Definir y entrenar el modelo ARIMA con hiperparámetros fijos
        best_params_arima = (2, 0, 2)
        modelo_arima = ARIMA(y_entrenamiento_30d, order=best_params_arima).fit()

        # Guardar el modelo
        os.makedirs("modelos", exist_ok=True)
        ruta_modelo_arima = "modelos/ARIMA.joblib"
        joblib.dump(modelo_arima, ruta_modelo_arima)
        print(f"Modelo ARIMA reentrenado y guardado en: {ruta_modelo_arima}")

        # Cargar y evaluar con datos nuevos
        parquet_file_path_nuevos_30d = get_path('data', 'nuevos', 'data_30d.parquet') # Para evaluar
        df_nuevos_30d = cargar_datos_30d(parquet_file_path_nuevos_30d) # OJO: Cambiado a 'nuevos'
        y_nuevos_30d = df_nuevos_30d["pct_cambio_30d"]  # Usar el target
        pred = modelo_arima.predict(start=y_nuevos_30d.index[0], end=y_nuevos_30d.index[-1])
        metricas_arima = evaluar_modelo_30d(y_nuevos_30d, pred)
        print("Métricas ARIMA en datos nuevos:", metricas_arima)

        if reentrenar_final:
            print("\nReentrenando el modelo ARIMA final con todos los datos...")
            df_final_30d = pd.concat([df_entrenamiento_30d, df_nuevos_30d])
            y_final_30d = df_final_30d["pct_cambio_30d"]
            modelo_arima_final = ARIMA(y_final_30d, order=best_params_arima).fit()
            joblib.dump(modelo_arima_final, ruta_modelo_arima.replace(".joblib", "_final.joblib"))
            print("Modelo ARIMA final reentrenado y guardado.")
    else:
        raise ValueError(f"Técnica desconocida: {tecnica}. Debe ser 'svr' o 'arima'.")



if __name__ == "__main__":
    # Reentrenar y evaluar SVR, con reentrenamiento final opcional
    reentrenar_y_evaluar_modelo(tecnica="svr", reentrenar_final=True)

    # Reentrenar y evaluar ARIMA, con reentrenamiento final opcional
    reentrenar_y_evaluar_modelo(tecnica="arima", reentrenar_final=True)
