import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Función para cargar datos desde un archivo Parquet
def cargar_datos(ruta_parquet):
    """
    Carga los datos desde un archivo Parquet y los devuelve como un DataFrame.
    :param ruta_parquet: Ruta del archivo Parquet.
    :return: DataFrame de pandas con los datos.
    """
    df = pd.read_parquet(ruta_parquet)
    return df

# Función para preprocesar los datos (elimina columnas irrelevantes y escala)
def preprocesar_datos(df):
    """
    Preprocesa el dataset eliminando columnas irrelevantes y escalando los datos.
    :param df: DataFrame original.
    :return: X (features), y (target) escalados y el scaler.
    """
    # Eliminar columnas no necesarias
    df = df.drop(columns=['pct_cambio_7d', 'pct_cambio_30d', 'tendencia_futura_30d'], errors='ignore')

    # Separar variables independientes (X) y variable objetivo (y)
    X = df.drop(columns=['tendencia_futura_7d'])
    y = df['tendencia_futura_7d']

    # Escalado de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Función para dividir los datos en train, val y test
def dividir_datos(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
    :param X: Features escaladas.
    :param y: Target.
    :param test_size: Proporción de datos de prueba.
    :param val_size: Proporción de datos de validación.
    :param random_state: Semilla aleatoria.
    :return: Conjuntos X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Función para buscar los mejores hiperparámetros para SVC
def buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val):
    """
    Realiza una búsqueda de hiperparámetros óptimos para SVC usando GridSearchCV.
    :param X_train: Datos de entrenamiento.
    :param y_train: Etiquetas de entrenamiento.
    :param X_val: Datos de validación.
    :param y_val: Etiquetas de validación.
    :return: Mejor modelo SVC entrenado y los mejores hiperparámetros.
    """
    parametros = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    }

    svc = SVC()
    grid_search = GridSearchCV(svc, parametros, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_
    mejores_parametros = grid_search.best_params_

    print("Mejores Hiperparámetros:", mejores_parametros)

    # Evaluar el modelo en el conjunto de validación
    y_val_pred = mejor_modelo.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(f"Exactitud en validación: {val_accuracy:.4f}")
    print(classification_report(y_val, y_val_pred))

    return mejor_modelo, mejores_parametros

# Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo entrenado en el conjunto de prueba.
    :param modelo: Modelo entrenado.
    :param X_test: Datos de prueba.
    :param y_test: Etiquetas de prueba.
    """
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Exactitud en prueba: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

# Función principal
def main():
    """
    Función principal que ejecuta el flujo completo del modelo.
    """
    ruta_parquet = "ruta/a/tu/dataset.parquet"  # Cambia por la ruta correcta
    df = cargar_datos(ruta_parquet)

    # Preprocesamiento de datos
    X, y, scaler = preprocesar_datos(df)

    # División en entrenamiento, validación y prueba
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)

    # Buscar mejores hiperparámetros y entrenar el modelo óptimo
    mejor_modelo, mejores_parametros = buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val)

    # Evaluar el modelo con los mejores hiperparámetros en el conjunto de prueba
    evaluar_modelo(mejor_modelo, X_test, y_test)

if __name__ == "__main__":
    main()
