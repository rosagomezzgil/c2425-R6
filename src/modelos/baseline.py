import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from typing import Tuple

def baseline_persistencia_categorica(
        df: pd.DataFrame,
        target_col: str,
        horizon: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implementa un modelo baseline de persistencia para variables categóricas.
    Predice que la tendencia será igual a la de `horizon` días atrás, permitiendo solo horizontes de 7 o 30 días.

    :param df: DataFrame con los datos de mercado.
    :type df: pd.DataFrame
    :param target_col: Nombre de la columna objetivo (valores posibles: -1 para bajista, 0 para neutra, 1 para alcista).
    :type target_col: str
    :param horizon: Horizonte de predicción (solo acepta valores 7 o 30 días).
    :type horizon: int
    :raises ValueError: Si el horizonte de predicción no es 7 o 30 días.
    :return: DataFrame con las predicciones añadidas y matriz de confusión del baseline.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    if horizon not in [7, 30]:
        raise ValueError("El horizonte de predicción solo puede ser 7 o 30 días.")

    df = df.copy()
    df["baseline_pred"] = df[target_col].shift(horizon)
    df = df.dropna()  # Elimina los valores nulos creados por el shift
    cm = pd.DataFrame(confusion_matrix(df[target_col], df["baseline_pred"], labels=[-1, 0, 1]),
                      index=["Real -1", "Real 0", "Real 1"],
                      columns=["Pred -1", "Pred 0", "Pred 1"])
    return df, cm


def baseline_persistencia_numerica(
        df: pd.DataFrame,
        target_col: str,
        horizon: int
) -> Tuple[pd.DataFrame, float, float]:
    """
    Implementa un modelo baseline de persistencia para variables numéricas.
    Predice que el valor de la variable será igual al de `horizon` días atrás.

    :param df: DataFrame con los datos de mercado.
    :type df: pd.DataFrame
    :param target_col: Nombre de la columna objetivo numérica.
    :type target_col: str
    :param horizon: Horizonte de predicción (solo acepta valores 7 o 30 días).
    :type horizon: int
    :raises ValueError: Si el horizonte de predicción no es 7 o 30 días.
    :return: DataFrame con las predicciones añadidas, MAE y RMSE del baseline.
    :rtype: Tuple[pd.DataFrame, float, float]
    """
    if horizon not in [7, 30]:
        raise ValueError("El horizonte de predicción solo puede ser 7 o 30 días.")

    df = df.copy()
    df["baseline_pred"] = df[target_col].shift(horizon)
    df = df.dropna()  # Elimina valores nulos creados por el shift
    mae = mean_absolute_error(df[target_col], df["baseline_pred"])
    rmse = mean_squared_error(df[target_col], df["baseline_pred"], squared=False)
    return df, mae, rmse