import pandas as pd
import numpy as np
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
def train(x, y):
    # Dividir los datos en conjunto de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Crear el modelo de árbol de decisión para regresión
    model = DecisionTreeRegressor(max_depth=5, random_state=42)

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
    csv_file = r'../../data/limpieza/cleaned_data.csv'  # Asegúrate de tener este archivo

    df = load_data(csv_file)
    x, y = preprocess_data(df)

    model, x_test, y_test = train(x, y)
    predict_and_plot(model, x_test, y_test)

if __name__ == "__main__":
    main()
