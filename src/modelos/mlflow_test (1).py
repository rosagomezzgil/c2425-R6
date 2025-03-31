"""
Instalación
    > pip install mlflow  

Ejecutar servidor local (por defecto escucha en http://localhost:5000):
    > mlflow server        
"""

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature

# Conexión al servidor
# MLFlow puede trabajar con muchos backends distintos
mlflow.set_tracking_uri(
    uri="http://127.0.0.1:5000"
)  # servidor local y archivos de texto
# mlflow.set_tracking_uri("sqlite:///mlruns.db") # base de datos SQLite

# Crear un experimento MLFlow. Un experimento consiste en varios "runs" y cada uno
# de ellos registra una configuración de hiperparámetros
mlflow.set_experiment("MLflow Quickstart")

# Cargar el dataset de Iris
X, y = datasets.load_iris(return_X_y=True)

# Partición en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hiperparámetros
params = {"solver": "lbfgs", "max_iter": 1000, "random_state": 8888}

# Entrenamiento del modelo
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Medición del error
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Empezar un "run" de MLFlow nuevo
with mlflow.start_run():
    # Registrar hiperparámetros
    mlflow.log_params(params)

    # Registrar el error
    mlflow.log_metric("accuracy", accuracy)

    # Establecer una etiqueta con información sobre el "run"
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Registrar el modelo y el conjunto de datos de entrenamiento.
    # Esto puede ocupar mucho espacio. Pensar si en vuestro proyecto merece la pena.
    signature = infer_signature(X_train, lr.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )


# ------------------------------------------------------------------------------

# Cargar un modelo registrado previamente en MLFlow
# Se puede cargar desde la URL del modelo, con nombre del modelo, con el ID del run, etc.
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# run_id = 'e1e920f48133445e8e41e51bb799bb94'
# loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/iris_model')

# model_name = 'tracking-quickstart'
# version = 1
# loaded_model = mlflow.pyfunc.load_model(f'models:/{model_name}/{version}')

# Comprobar que funciona
predictions = loaded_model.predict(X_test)
assert (predictions == y_pred).all()
