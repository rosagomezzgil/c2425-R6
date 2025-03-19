import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 1. Cargar los datos
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])  # Asegurar que la fecha esté en formato datetime
    df.set_index('date', inplace=True)
    return df


# 2. Preprocesar los datos
def preprocess_data(df, horizonte):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['close_binance']])

    X, y = [], []
    for i in range(len(df_scaled) - horizonte):
        X.append(df_scaled[i:i + horizonte])
        y.append(df_scaled[i + horizonte])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# 3. Crear y entrenar el modelo LSTM
def train_lstm(X_train, y_train, horizonte):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(horizonte, 1)),
        Dropout(0.2),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    return model


# 4. Hacer predicciones
def predict_and_plot(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Desescalar
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Real')
    plt.plot(predictions, label='Predicción')
    plt.legend()
    plt.show()


# 5. Ejecutar el flujo
def main():
    csv_file = 'bitcoin_data.csv'  # Asegúrate de tener este archivo
    horizonte = 10  # Número de días a predecir

    df = load_data(csv_file)
    X, y, scaler = preprocess_data(df, horizonte)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = train_lstm(X_train, y_train, horizonte)
    predict_and_plot(model, X_test, y_test, scaler)


if __name__ == "__main__":
    main()