from src.modelos.B_preprocesamiento import preprocesar_datos_30d
from src.modelos.E_ajuste_hiperparametros import validar_modelo_timeseries
from src.modelos.F_metricas import evaluar_modelo_30d
from src.modelos.A_cargar_datos import cargar_datos_30d
from src.modelos.D_division_temporal import dividir_datos_temporalmente_30d

import mlflow
from statsmodels.tsa.arima.model import ARIMA


def main_arima(parquet_path: str, target_col: str = 'pct_cambio_30d'):
    df = cargar_datos_30d(parquet_path)
    df = preprocesar_datos_30d(df, target_col)

    # ✅ Dividir datos en entrenamiento y test antes de ajustar
    y_train, y_test = dividir_datos_temporalmente_30d(df, target_col, test_ratio=0.2)

    param_grid = [(1, 1, 1), (2, 1, 2), (1, 0, 1), (2, 0, 2)]

    mlflow.set_tracking_uri("file:../../mlruns")
    mlflow.set_experiment("ARIMA con TimeSeriesSplit")

    with mlflow.start_run():

        model, best_order = validar_modelo_timeseries(y_train, param_grid, ARIMA)
        pred = model.predict(start=y_test.index[0], end=y_test.index[-1])
        mse, mae, rmse, r2, acc = evaluar_modelo_30d(y_test, pred)

        mlflow.log_params({"order": best_order})
        mlflow.log_metrics({
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "acc": acc
        })


if __name__ == "__main__":
    main_arima("../../../data/parquet/data_30d.parquet")
