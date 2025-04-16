from sklearn.svm import SVR
from src.modelos.E_ajuste_hiperparametros import ajustar_modelo  # Función de ajuste de hiperparámetros
from src.modelos.A_cargar_datos import cargar_datos_7d  # Función de carga de datos
from src.modelos.B_preprocesamiento import preprocesar_datos_7d  # Función de preprocesamiento
from src.modelos.D_division_temporal import dividir_datos_temporalmente_7d  # Función para dividir los datos temporalmente
from src.modelos.F_metricas import evaluar_modelo_7d  # Función de evaluación del modelo
from src.modelos.C_seleccion_variables import seleccionar_variables
from src.modelos.G_graficar_predicciones import plot_predictions
import mlflow


def main():
    # Configuración de mlflow
    mlflow.set_tracking_uri("file:../../mlruns")
    mlflow.set_experiment("Modelo SVR")

    # Cargar y preprocesar los datos
    df = cargar_datos_7d("../../../data/parquet/data_7d.parquet")
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
    with mlflow.start_run():
        best_model, best_params = ajustar_modelo(svr_model, X_train_val, y_train_val, param_grid, strategy="random")

        # Evaluación del modelo
        mse, mae, rmse, r2, acc = evaluar_modelo_7d(best_model, X_test, y_test)

        fig = plot_predictions(y_test, modelo=best_model, X=X_test)

        # Loguear los mejores parámetros y métricas
        mlflow.log_params(best_params)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})
        mlflow.log_figure(fig, "prediction_plot.png")


if __name__ == "__main__":
    main()
