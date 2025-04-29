from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def preprocesar_datos_7d(df, tecnica, target_col, scaler = None):
    """
    Preprocesa los datos:
    - Separa X e y
    - Escala si es necesario
    - Si scaler es None, lo crea y hace fit_transform (entrenamiento)
    - Si scaler ya existe, solo hace transform (predicción)

    :param df: DataFrame
    :param tecnica: 'svr', 'lstm', 'arbol', etc.
    :param target_col: nombre de la columna objetivo
    :param scaler: scaler existente o None
    :return: X procesado, y, scaler
    """

    df = df.copy()
    df = df.dropna()

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if tecnica in ["svr", "lstm"]:
        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)  # Entrenamiento
        else:
            X = scaler.transform(X)  # Predicción
    return X, y, scaler

def preprocesar_datos_30d(df: pd.DataFrame, target_col: str):

    df = df.copy()
    df = df.dropna()

    df.drop_duplicates(inplace=True)
    
    adf_result = adfuller(df[target_col])
    if adf_result[1] > 0.05:
        print("La serie no es estacionaria, se aplicará diferenciación")
        df[target_col] = df[target_col].diff().dropna()
        df = df.dropna()

    return df