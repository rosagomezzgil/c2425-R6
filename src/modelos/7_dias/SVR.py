from datetime import datetime
from sklearn.svm import SVR
from utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d  # Función de carga de datos
from utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_7d  # Función de preprocesamiento
from utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_7d  # Función para dividir los datos temporalmente
from utils.modelos_sklearn.E_ajuste_hiperparametros import ajustar_modelo  # Función de ajuste de hiperparámetros
from utils.modelos_sklearn.F_metricas import evaluar_modelo_7d  # Función de evaluación del modelo
from utils.modelos_sklearn.C_seleccion_variables import seleccionar_variables
from utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from utils.mlflow_utils import setup_mlflow
from utils.paths import get_path
import mlflow


def main():
    # Configuración de mlflow
    setup_mlflow("Modelo SVR")

    # Cargar y preprocesar los datos
    df = cargar_datos_7d(get_path('data', 'parquet', 'data_7d.parquet'))
    X, y, scaler = preprocesar_datos_7d(df, tecnica="svr", target_col="pct_cambio_7d")

    # Seleccion de variables
    X, selector = seleccionar_variables(X, y, tecnica="svr")

    # Dividir los datos en conjunto de entrenamiento y test
    X_train_val, y_train_val, X_test, y_test = dividir_datos_temporalmente_7d(X, y)

    # Definir el modelo SVR
    svr_model = SVR()

    # Definir los hiperparámetros para buscar
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2],
        'kernel': ['linear', 'rbf']
    }

    # Ajustar el modelo con los mejores hiperparámetros usando tu función de ajuste
    hora_actual = datetime.now().strftime("%H-%M-%S")
    with mlflow.start_run(run_name=f"SVR-{hora_actual}"):
        best_model, best_params = ajustar_modelo(svr_model, X_train_val, y_train_val, param_grid, strategy="random")

        # Evaluación del modelo
        mse, mae, rmse, r2, acc = evaluar_modelo_7d(best_model, X_test, y_test)

        fig = plot_predictions(y_test, modelo=best_model, X=X_test)

        # Loguear los mejores parámetros y métricas
        mlflow.log_params(best_params)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})
        mlflow.log_figure(fig, "prediction_plot.png")

        #Reentrenar el modelo con train_val completo
        modelo_reentrenado = SVR(**best_params)
        modelo_reentrenado.fit(X_train_val, y_train_val)

        # Evaluar el modelo reducido
        mse2, mae2, rmse2, r2_2, acc2 = evaluar_modelo_7d(
            modelo_reentrenado, X_test, y_test
        )
        fig2 = plot_predictions(y_test, modelo=best_model, X=X_test)

        mlflow.log_metrics({"mse": mse2, "mae": mae2, "rmse": rmse2, "r2": r2_2, "acc": acc2})
        mlflow.log_figure(fig2, "prediction_plot.png")

if __name__ == "__main__":
    main()
