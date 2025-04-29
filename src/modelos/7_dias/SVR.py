from utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d  # Función de carga de datos
from utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_7d  # Función para dividir los datos temporalmente
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from utils.modelos_sklearn.E_ajuste_hiperparametros import ajustar_modelo
from utils.modelos_sklearn.C_seleccion_variables import seleccionar_variables
from utils.modelos_sklearn.F_metricas import evaluar_modelo_7d  # Función de evaluación del modelo
from utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from utils.mlflow_utils import setup_mlflow
from utils.paths import get_path
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from datetime import datetime

def main():
    # Configuración de mlflow
    setup_mlflow("Modelo SVR Mejorado - Abordando Sobreajuste")

    # Cargar y preprocesar los datos
    df = cargar_datos_7d(get_path('data', 'parquet', 'data_7d.parquet'))
    df.dropna(inplace=True)
    y = df["pct_cambio_7d"]
    X_original = df.drop(columns=["pct_cambio_7d"])

    # Definir los valores a probar para las estrategias contra el sobreajuste
    k_vars_a_probar = [5, 10, min(15, X_original.shape[1])]
    c_values_a_probar = [0.01, 0.1, 1, 10]
    epsilon_values_a_probar = [0.1, 0.2, 0.5]
    kernels_a_probar = ['rbf', 'linear']

    for k_vars in k_vars_a_probar:
        # Selección de variables con el k actual
        conjuntos_X = seleccionar_variables(X_original, y, tecnica="svr", k_vars=k_vars)

        for nombre_conjunto, X_seleccionado in conjuntos_X.items():
            print(f"\nEntrenando y evaluando con el conjunto de variables: {nombre_conjunto}")

            # Escalar las características seleccionadas
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_seleccionado)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_seleccionado.columns, index=X_seleccionado.index)

            # Dividir los datos en conjunto de entrenamiento y test
            X_train_val, y_train_val, X_test_scaled, y_test = dividir_datos_temporalmente_7d(X_scaled_df, y)

            # Definir el modelo SVR
            svr_model = SVR()

            # Definir el espacio de hiperparámetros a probar
            param_grid_svr = {
                'C': c_values_a_probar,
                'epsilon': epsilon_values_a_probar,
                'kernel': kernels_a_probar
                # 'gamma': ['scale', 'auto'] # Puedes mantener gamma o probar valores específicos
            }

            # Ajustar el modelo con los mejores hiperparámetros usando la función de ajuste mejorada
            hora_actual = datetime.now().strftime("%H-%M-%S")
            with mlflow.start_run(run_name=f"SVR-f_regression-k{k_vars}-{hora_actual}"):
                best_model_svr, best_params_svr = ajustar_modelo(
                    svr_model, X_train_val, y_train_val, param_grid_svr, strategy="random", scoring="r2", n_iter=50
                )

                # Escalar el conjunto de prueba usando el scaler ajustado en el entrenamiento
                X_test = scaler.transform(X_test_scaled)

                # Evaluación del modelo
                mse_svr, mae_svr, rmse_svr, r2_svr, acc_svr = evaluar_modelo_7d(best_model_svr, X_test, y_test)

                fig_svr = plot_predictions(y_test, modelo=best_model_svr, X=X_test)

                # Loguear los mejores parámetros y métricas para SVR
                mlflow.log_params(best_params_svr)
                mlflow.log_metrics({"mse": mse_svr, "mae": mae_svr, "rmse": rmse_svr, "r2": r2_svr, "acc": acc_svr, "k_vars": k_vars})
                mlflow.log_figure(fig_svr, f"prediction_plot_svr_f_regression_k{k_vars}.png")

                # Reentrenar el modelo con train_val completo (escalando primero)
                X_train_val_scaled = scaler.fit_transform(X_train_val)
                modelo_reentrenado_svr = SVR(**best_params_svr)
                modelo_reentrenado_svr.fit(X_train_val_scaled, y_train_val)

                # Evaluar el modelo reentrenado
                X_test_scaled_reentrenado = scaler.transform(X_test_scaled)
                mse2_svr, mae2_svr, rmse2_svr, r2_2_svr, acc2_svr = evaluar_modelo_7d(
                    modelo_reentrenado_svr, X_test_scaled_reentrenado, y_test
                )
                fig2_svr = plot_predictions(y_test, modelo=modelo_reentrenado_svr, X=X_test_scaled_reentrenado)

                mlflow.log_metrics({"mse_retrain": mse2_svr, "mae_retrain": mae2_svr, "rmse_retrain": rmse2_svr, "r2_retrain": r2_2_svr, "acc_retrain": acc2_svr, "k_vars": k_vars})
                mlflow.log_figure(fig2_svr, f"prediction_plot_svr_f_regression_k{k_vars}_retrain.png")


if __name__ == "__main__":
    main()