from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def preprocesar_datos_7d(df, tecnica, target_col):
    """
    Preprocesa los datos según la técnica:
    - Separa X e y
    - Escala si es necesario

    :param df: DataFrame ya limpio (ya contiene solo variables relevantes)
    :param tecnica: 'svr', 'lstm', 'arbol', etc.
    :param target_col: nombre de la columna objetivo
    :return: X, y, scaler o None
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    scaler = None
    if tecnica in ["svr", "lstm"]:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, scaler


def preprocesar_datos_30d(df: pd.DataFrame, target_col: str):
    df = df.copy()
    df = df.dropna()

    adf_result = adfuller(df[target_col])
    if adf_result[1] > 0.05:
        print("La serie no es estacionaria, se aplicará diferenciación")
        df[target_col] = df[target_col].diff().dropna()
        df = df.dropna()

    return df