import mlflow
from sklearn.tree import DecisionTreeRegressor
from src.modelos.A_cargar_datos import cargar_datos_7d
from src.modelos.B_preprocesamiento import preprocesar_datos_7d
from src.modelos.D_division_temporal import dividir_datos_temporalmente_7d
from src.modelos.E_ajuste_hiperparametros import ajustar_modelo
from src.modelos.F_metricas import evaluar_modelo_7d

def main():
    mlflow.set_tracking_uri(r"file:../../mlruns")
    mlflow.set_experiment("Modelo Arbol")

    df = cargar_datos_7d("../../../data/limpieza/cleaned_data.csv")
    X, y, _ = preprocesar_datos_7d(df, tecnica="arbol", target_col="pct_cambio_7d")

    # Cambiado: ahora separamos solo train_val y test
    X_train_val, y_train_val, X_test, y_test = dividir_datos_temporalmente_7d(X, y)

    modelo = DecisionTreeRegressor(random_state=42)
    param_grid = {
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5],
    }

    try:
        with mlflow.start_run():
            best_model, best_params = ajustar_modelo(
                modelo, X_train_val, y_train_val, param_grid
            )
            mse, mae, rmse, r2, acc = evaluar_modelo_7d(best_model, X_test, y_test)

            mlflow.log_params(best_params)
            mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()