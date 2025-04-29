from src.utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d, cargar_datos_30d
from src.utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_7d, preprocesar_datos_30d
from utils.modelos_sklearn.F_metricas import evaluar_modelo_7d, evaluar_modelo_30d
from src.utils.paths import get_path
import os
import joblib
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA


def reentrenar_modelo(tecnica):

    if tecnica == "svr":
        # Cargar todos los datos disponibles
        df = cargar_datos_7d(get_path('data', 'parquet', 'data_7d.parquet'))
        X, y, scaler = preprocesar_datos_7d(df, tecnica=tecnica, target_col="pct_cambio_7d")

        # Cargar los mejores hiperparámetros
        best_params = {'kernel': 'rbf', 'epsilon': 0.2, 'C': 0.1}

        # Reentenar el modelo
        modelo = SVR(**best_params)
        modelo.fit(X, y)
    else:
        #Cargar todos los datos disponibles
        df = cargar_datos_30d(get_path('data', 'parquet', 'data_30d.parquet'))
        df = preprocesar_datos_30d(df, "pct_cambio_30d")

        #Cargar los mejores hiperparámetros
        best_params = (1, 0, 1)

        #Reentrenar el modelo
        modelo = ARIMA(df["pct_cambio_30d"], order=best_params)
        modelo = modelo.fit()

    # Guardar localmente
    os.makedirs("modelos_entrenados", exist_ok=True)
    ruta_modelo = f"modelos_entrenados/{tecnica}_modelo.joblib"
    joblib.dump(modelo, ruta_modelo)

    # Guardar el escalado
    if tecnica == "svr":
        ruta_scaler = "modelos_entrenados/svr_scaler.joblib"
        joblib.dump(scaler, ruta_scaler)
    else:
        ruta_scaler = None

    return modelo, ruta_modelo, ruta_scaler

def evaluar_con_datos_nuevos(ruta_modelo,target, tecnica, ruta_scaler=None):

    # Cargar el modelo desde el archivo
    modelo = joblib.load(ruta_modelo)

    if tecnica == "svr":
        #Cargar datos nuevos
        scaler = joblib.load(ruta_scaler)
        df = cargar_datos_7d(get_path('data', 'nuevos', 'data_7d.parquet'))
        X_new, y_new, _ = preprocesar_datos_7d(df, tecnica=tecnica, target_col=target, scaler=scaler)
        return evaluar_modelo_7d(modelo, X_new, y_new)

    else:
        df = cargar_datos_30d(get_path('data', 'parquet', 'data_30d.parquet'))
        df = preprocesar_datos_30d(df, target)
        df = df[target]
        pred = modelo.predict(start=df.index[0], end=df.index[-1])
        return evaluar_modelo_30d(df, pred)


if __name__ == "__main__":

    #Reentrenar y evaluar SVR
    modelo_svr, ruta_modelo_svr, ruta_scaler = reentrenar_modelo(
        tecnica= "svr",
    )

    evaluar_con_datos_nuevos(ruta_modelo=ruta_modelo_svr, target= "pct_cambio_7d", tecnica="svr",ruta_scaler=ruta_scaler)

    #Reentrenar y evaluar ARIMA
    modelo, ruta_modelo_arima, _ = reentrenar_modelo(
        tecnica="arima",
    )

    evaluar_con_datos_nuevos(ruta_modelo=ruta_modelo_arima, target="pct_cambio_30d", tecnica="arima")