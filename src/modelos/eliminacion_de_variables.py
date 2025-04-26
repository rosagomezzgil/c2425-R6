import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

def seleccion_variables_suave(df, target, umbral_corr=0.05, umbral_colinealidad=0.9, usar_modelo=True, random_state=42):
    """
    Selecciona variables de forma suave:
    1. Filtra por correlación con el target.
    2. Elimina variables altamente correlacionadas entre sí.
    3. (Opcional) Ordena por importancia de modelo LightGBM.

    Args:
        df (pd.DataFrame): DataFrame original.
        target (str): Nombre de la variable objetivo.
        umbral_corr (float): Umbral mínimo de correlación absoluta para conservar variables.
        umbral_colinealidad (float): Umbral máximo de correlación entre features para eliminar redundantes.
        usar_modelo (bool): Si True, ordena las variables finales por importancia de LightGBM.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        list: Variables seleccionadas (ordenadas si usar_modelo=True).
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # 1. Filtrar por correlación con el target
    correlaciones = df.corr()[target].drop(target)
    selected = correlaciones[correlaciones.abs() > umbral_corr].index.tolist()

    # 2. Eliminar variables altamente correlacionadas entre sí
    corr_matrix = df[selected].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > umbral_colinealidad)]
    selected = [col for col in selected if col not in to_drop]

    # 3. (Opcional) Ordenar por importancia de modelo
    if usar_modelo and selected:
        model = LGBMRegressor(random_state=random_state, verbose=-1)
        model.fit(df[selected], df[target])
        importances = model.feature_importances_
        selected = [feat for _, feat in sorted(zip(importances, selected), reverse=True)]

    return selected
