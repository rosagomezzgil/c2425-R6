import mlflow
import numpy as np
from datetime import datetime
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d
from src.utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_7d
from src.utils.modelos_sklearn.F_metricas import evaluar_modelo_7d
from src.utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from src.utils.mlflow_utils import setup_mlflow
from src.utils.paths import get_path

from src.modelos.arquitecturas.lstm_model import SklearnLikeLSTM  # Clase adaptada


def crear_ventanas(X, y, window_size=10):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i + window_size])
        ys.append(y.iloc[i + window_size])
    return np.array(Xs), np.array(ys)


def preparar_datos():
    df = cargar_datos_7d(get_path('data', 'parquet', 'data_7d.parquet'))
    X, y, scaler = preprocesar_datos_7d(df, tecnica="lstm", target_col="pct_cambio_7d")
    X_seq, y_seq = crear_ventanas(X, y)

    n_train = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:n_train], X_seq[n_train:]
    y_train, y_test = y_seq[:n_train], y_seq[n_train:]

    return X_train, y_train, X_test, y_test


def main():
    setup_mlflow("LSTM-PyTorch")

    X_train, y_train, X_test, y_test = preparar_datos()
    input_shape = X_train.shape[1:]  # (seq_len, n_features)

    param_grid = {
        "hidden_size": [50, 100],
        "epochs": [10, 20],
        "batch_size": [16, 32],
        "lr": [0.001],
        "input_shape": [input_shape],  # Necesario para instanciar correctamente
    }

    tscv = TimeSeriesSplit(n_splits=3)

    grid = GridSearchCV(
        estimator=SklearnLikeLSTM(),
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=tscv,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    # Evaluación en test
    y_pred = best_model.predict(X_test)
    mse, mae, rmse, r2, acc = evaluar_modelo_7d(best_model, X_test, y_test)

    plot_predictions(y_test, y_pred)

    hora_actual = datetime.now().strftime("%H-%M-%S")
    with mlflow.start_run(run_name=f"LSTM-SklearnLike-{hora_actual}"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})


if __name__ == "__main__":
    main()
