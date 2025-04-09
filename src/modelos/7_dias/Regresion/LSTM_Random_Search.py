import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.random import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# Fijar semillas
def fijar_semillas(seed: int = 123):
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# 1. Cargar los datos
def load_data(parquet_file: str):
    df = pd.read_parquet(parquet_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# 2. Preprocesar los datos
def preprocess_data(df: pd.DataFrame, horizonte: int, target_col: str, sequence_length: int = 30):
    assert target_col in df.columns, f"La columna {target_col} no está en el DataFrame"
    assert target_col == f'pct_cambio_{horizonte}d', "El nombre de la columna objetivo no coincide con el horizonte"
    col_categorica = f'tendencia_futura_{horizonte}d'
    assert col_categorica in df.columns, f"La columna {col_categorica} no está en el DataFrame"

    features = df.drop(columns=[target_col, col_categorica])
    y = df[target_col].values

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns)

    X = []
    for i in range(len(features_scaled) - sequence_length - horizonte + 1):
        X.append(features_scaled.iloc[i: i + sequence_length].values)

    X = np.array(X)
    return X, y[: len(X)], scaler

def split_data(X, y, test_size=0.1, val_size=0.1, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_relative_size, shuffle=shuffle)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Modelo base para wrapper
def build_model(units=50):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# 3. Crear y entrenar el modelo LSTM
def train_lstm(X_train, y_train, units=50, epochs=50, batch_size=32):
    print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
    print("Dispositivo en uso:", tf.test.gpu_device_name())

    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    return model

def predict_and_plot(model, X_test, y_test, target_col):
    predictions = model.predict(X_test).flatten()
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Real", color="blue")
    plt.plot(predictions, label="Predicción", color="red", linestyle="dashed")
    plt.legend()
    plt.xlabel("Días")
    plt.ylabel(f"Valor de {target_col}")
    plt.title(f"Predicción vs Real: {target_col}")
    plt.show()

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    pred_sign = np.sign(predictions)
    real_sign = np.sign(y_test)
    directional_acc = accuracy_score(real_sign, pred_sign)

    print("\nMétricas del modelo:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy Direccional: {directional_acc:.4f}")

    return mse, mae, rmse, r2, directional_acc

# NUEVA FUNCION: Random Search
def optimize_hyperparameters(X_train, y_train, X_val, y_val, param_grid, n_iter=10):
    keras_reg = KerasRegressor(build_fn=build_model, verbose=0)

    # Combinar train y val para usar cross-validation
    X_combined = np.concatenate((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))

    random_search = RandomizedSearchCV(
        estimator=keras_reg,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=2,
        random_state=42
    )

    random_search.fit(X_combined, y_combined)

    best_params = random_search.best_params_
    print("\nMejores hiperparámetros encontrados:")
    print(best_params)
    return best_params

# 5. Ejecutar flujo principal
def main():
    mlflow.set_tracking_uri(r"file:../../mlruns")
    mlflow.set_experiment("LSTM Optimization 7d")

    param_grid = {
        "units": [32, 64, 128],
        "epochs": [30, 50, 100, 500],
        "batch_size": [16, 32, 64],
    }

    parquet_file = "../../../../data/parquet/data_7d.parquet"
    horizonte = 7
    target_col = "pct_cambio_7d"
    sequence_length = 7

    fijar_semillas()
    df = load_data(parquet_file)
    X, y, scaler = preprocess_data(df, horizonte, target_col, sequence_length)
    global X_train, X_val  # Necesario para que build_model vea los shapes
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 0.1, 0.1, shuffle=False)
    print(f"Datos preparados para sequence_length: {sequence_length}\n")

    # Optimización con Random Search
    best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, param_grid, n_iter=10)

    with mlflow.start_run(run_name=f"LSTM_seq{sequence_length}"):
        best_model = train_lstm(
            X_train, y_train,
            units=best_params['units'],
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size']
        )

        mse, mae, rmse, r2, directional_acc = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(best_params)
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "directional_acc": directional_acc})

    predict_and_plot(best_model, X_test, y_test, target_col)

if __name__ == "__main__":
    main()
