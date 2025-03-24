import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


# Cargar los datos
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df


# Separar características (X) y etiquetas (y)
def preprocess_data(df):
    x = df.iloc[:, :-4]  # Asegúrate de seleccionar correctamente las columnas de características
    y = df['pct_cambio_7d']  # Variable objetivo para regresión
    return x, y


# Crear y entrenar el modelo de regresión
def train(x, y, max_depth):
    # Dividir los datos en conjunto de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Crear el modelo de árbol de decisión para regresión
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

    # Entrenar el modelo
    model.fit(x_train, y_train)

    return model, x_test, y_test


# Hacer predicciones y visualizar
def predict_and_plot(model, x_test, y_test):
    predictions = model.predict(x_test)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='Real')
    plt.plot(predictions, label='Predicción', linestyle='dashed')
    plt.legend()
    plt.xlabel("Índice de muestra")
    plt.ylabel("Valor Predicho")
    plt.title("Comparación entre valores reales y predichos")
    plt.show()


# Ejecutar el flujo
def main():
    parser = argparse.ArgumentParser(description="Entrenar y evaluar un modelo de regresión con DecisionTreeRegressor")
    parser.add_argument("--csv_file", type=str, required=True, help="Ruta del archivo CSV de entrada")
    parser.add_argument("--max_depth", type=int, default=5, help="Profundidad máxima del árbol de decisión")

    args = parser.parse_args()

    df = load_data(args.csv_file)
    x, y = preprocess_data(df)

    model, x_test, y_test = train(x, y, args.max_depth)
    predict_and_plot(model, x_test, y_test)


if __name__ == "__main__":
    main()
