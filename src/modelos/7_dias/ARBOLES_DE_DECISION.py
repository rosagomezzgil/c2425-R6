import mlflow
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from src.modelos.A_cargar_datos import cargar_datos_7d
from src.modelos.B_preprocesamiento import preprocesar_datos_7d
from src.modelos.D_division_temporal import dividir_datos_temporalmente_7d
from src.modelos.E_ajuste_hiperparametros import ajustar_modelo
from src.modelos.F_metricas import evaluar_modelo_7d
from src.modelos.G_graficar_predicciones import plot_predictions
import pandas as pd

def analizar_importancias(modelo, X, umbral_importancia=0.01, top_n=20):
    """Muestra, guarda y retorna las variables más importantes según el árbol."""
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    importancias = importancias.sort_values(ascending=False)

    # Mostrar top n en consola
    print("\nTop variables por importancia:")
    print(importancias.head(top_n))
    print()

    # Seleccionar las variables importantes
    variables_importantes = importancias[importancias > umbral_importancia].index.tolist()

    return variables_importantes


def main():
    mlflow.set_tracking_uri(r"file:../../../mlruns")
    mlflow.set_experiment("Modelo Arbol")

    df = cargar_datos_7d("../../../data/parquet/data_7d.parquet")
    X, y, _ = preprocesar_datos_7d(df, tecnica="arbol", target_col="pct_cambio_7d")

    # Cambiado: ahora separamos solo train_val y test
    X_train_val, y_train_val, X_test, y_test = dividir_datos_temporalmente_7d(X, y)

    modelo = DecisionTreeRegressor(random_state=42)

    param_grid = {
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],  # Para controlar el número de variables evaluadas en cada split
    }

    try:
        hora_actual = datetime.now().strftime("%H-%M-%S")
        with mlflow.start_run(run_name=f"Arbol-{hora_actual}"):
            best_model, best_params = ajustar_modelo(
                modelo, X_train_val, y_train_val, param_grid
            )
            mse, mae, rmse, r2, acc = evaluar_modelo_7d(best_model, X_test, y_test)

            fig = plot_predictions(y_test, modelo=best_model, X=X_test)

            mlflow.log_params(best_params)
            mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})
            mlflow.log_figure(fig, "prediction_plot.png")

            #Seleccionar las variables importantes
            variables_importantes = analizar_importancias(best_model, X_train_val)

            # Reentrenamiento con solo variables importantes
            X_train_val_filtrado = X_train_val[variables_importantes]
            X_test_filtrado = X_test[variables_importantes]


            modelo_reentrenado = DecisionTreeRegressor(random_state=42, **best_params)
            modelo_reentrenado.fit(X_train_val_filtrado, y_train_val)

            # Evaluar el modelo reducido

            mse2, mae2, rmse2, r2_2, acc2 = evaluar_modelo_7d(
                modelo_reentrenado, X_test_filtrado, y_test
            )

            mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})
            mlflow.log_figure(fig, "prediction_plot.png")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()