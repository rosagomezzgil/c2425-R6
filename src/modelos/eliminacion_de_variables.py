# import numpy as np
# import pandas as pd
# from lightgbm import LGBMRegressor
#
# def seleccion_variables_suave(df, target, umbral_corr=0.05, umbral_colinealidad=0.9, usar_modelo=True, random_state=42):
#     """
#     Selecciona variables de forma suave:
#     1. Filtra por correlación con el target.
#     2. Elimina variables altamente correlacionadas entre sí.
#     3. (Opcional) Ordena por importancia de modelo LightGBM.
#
#     Args:
#         df (pd.DataFrame): DataFrame original.
#         target (str): Nombre de la variable objetivo.
#         umbral_corr (float): Umbral mínimo de correlación absoluta para conservar variables.
#         umbral_colinealidad (float): Umbral máximo de correlación entre features para eliminar redundantes.
#         usar_modelo (bool): Si True, ordena las variables finales por importancia de LightGBM.
#         random_state (int): Semilla para reproducibilidad.
#
#     Returns:
#         list: Variables seleccionadas (ordenadas si usar_modelo=True).
#     """
#     df = df.copy()
#     df['timestamp'] = pd.to_datetime(df['timestamp'])
#     df.set_index('timestamp', inplace=True)
#
#     # 1. Filtrar por correlación con el target
#     correlaciones = df.corr()[target].drop(target)
#     selected = correlaciones[correlaciones.abs() > umbral_corr].index.tolist()
#
#     # 2. Eliminar variables altamente correlacionadas entre sí
#     corr_matrix = df[selected].corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = [col for col in upper.columns if any(upper[col] > umbral_colinealidad)]
#     selected = [col for col in selected if col not in to_drop]
#
#     # 3. (Opcional) Ordenar por importancia de modelo
#     if usar_modelo and selected:
#         model = LGBMRegressor(random_state=random_state, verbose=-1)
#         model.fit(df[selected], df[target])
#         importances = model.feature_importances_
#         selected = [feat for _, feat in sorted(zip(importances, selected), reverse=True)]
#
#     return selected

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from typing import List, Dict, Union, Tuple, Optional


def seleccion_variables_suave(
        df: pd.DataFrame,
        target: str,
        umbral_corr: float = 0.05,
        umbral_colinealidad: float = 0.9,
        usar_modelo: bool = True,
        categorias_importantes: Optional[List[str]] = None,
        conservar_minimo_por_categoria: bool = True,
        pesos_categorias: Optional[Dict[str, float]] = None,
        retraso_temporal: Optional[int] = None,
        random_state: int = 42
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Selecciona variables para predecir Bitcoin con enfoque por categorías de datos.

    Args:
        df (pd.DataFrame): DataFrame con datos de Bitcoin y otras variables relacionadas.
        target (str): Nombre de la variable objetivo (ej: "tendencia_futura_7d").
        umbral_corr (float): Umbral mínimo de correlación absoluta para conservar variables.
        umbral_colinealidad (float): Umbral máximo de correlación entre features para eliminar redundantes.
        usar_modelo (bool): Si True, ordena las variables finales por importancia de LightGBM.
        categorias_importantes (List[str], opcional): Lista de sufijos/categorías que se deben priorizar.
        conservar_minimo_por_categoria (bool): Si True, conserva al menos una variable por categoría.
        pesos_categorias (Dict[str, float], opcional): Pesos para favorecer ciertas categorías.
        retraso_temporal (int, opcional): Si se proporciona, crea variables con retraso temporal.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        Tuple[List[str], Dict[str, List[str]]]:
            - Lista de variables seleccionadas
            - Diccionario con variables seleccionadas por categoría
    """
    # Asegurarse que el dataframe sea una copia
    df = df.copy()

    # Manejar timestamp si existe
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Identificar categorías/fuentes de datos automáticamente por sufijos
    categorias = {
        'binance': '_binance',
        'market_depth': '_mDbinance',
        'fear_greed': '_fg',
        'google_trends': '_gtrends',
        'hashrate': '_hashrate',
        'sp500': '_sp500',
        'nasdaq': '_nasdaq',
        'dxy': '_dxy'  # Índice del dólar
    }

    # Pesos por defecto para cada categoría si no se especifican
    if pesos_categorias is None:
        pesos_categorias = {
            'binance': 1.0,  # Datos directos de Bitcoin (prioridad alta)
            'market_depth': 0.9,  # Datos de profundidad de mercado (prioridad alta)
            'fear_greed': 0.8,  # Sentimiento del mercado (prioridad media-alta)
            'google_trends': 0.7,  # Interés público (prioridad media)
            'hashrate': 0.8,  # Datos de minería (prioridad media-alta)
            'sp500': 0.6,  # Correlación con mercados tradicionales (prioridad media-baja)
            'nasdaq': 0.6,  # Correlación con mercados tecnológicos (prioridad media-baja)
            'dxy': 0.5  # Correlación con el dólar (prioridad baja)
        }

    # Aplicar retraso temporal si se solicita
    if retraso_temporal is not None and retraso_temporal > 0:
        # Crear variables con retraso temporal para cada variable (excepto el target)
        variables = [col for col in df.columns if col != target]
        for col in variables:
            df[f"{col}_lag{retraso_temporal}"] = df[col].shift(retraso_temporal)

        # Eliminar filas con NaN creadas por el shift
        df.dropna(inplace=True)

    # 1. Obtener correlaciones con el target
    correlaciones = df.corr()[target].drop(target)

    # Variables seleccionadas y por categoría
    selected = []
    selected_by_category = {cat: [] for cat in categorias.keys()}

    # 2. Filtrar variables por correlación con el target y organizarlas por categoría
    for col in correlaciones.index:
        if abs(correlaciones[col]) > umbral_corr:
            for cat_name, suffix in categorias.items():
                if suffix in col:
                    selected.append(col)
                    selected_by_category[cat_name].append(col)
                    break

    # 3. Eliminar variables altamente correlacionadas entre sí, priorizando por categoría
    to_keep = []
    variables_por_categoria = {}

    # Primero organizar por categoría para proceso de selección más inteligente
    for cat_name, suffix in categorias.items():
        cat_vars = [var for var in selected if suffix in var]
        variables_por_categoria[cat_name] = cat_vars

    # Si hay categorías importantes, ordenarlas primero
    cat_order = list(categorias.keys())
    if categorias_importantes:
        for cat in reversed(categorias_importantes):
            if cat in cat_order:
                cat_order.remove(cat)
                cat_order.insert(0, cat)

    # Aplicar pesos a correlaciones para favorecer ciertas categorías
    weighted_corr = {}
    for col in selected:
        peso = 1.0
        for cat_name, suffix in categorias.items():
            if suffix in col:
                peso = pesos_categorias.get(cat_name, 1.0)
                break
        weighted_corr[col] = abs(correlaciones[col]) * peso

    # Ordenar variables por correlación ponderada
    selected_weighted = sorted(selected, key=lambda x: weighted_corr[x], reverse=True)

    # Eliminar colinealidad considerando la importancia de categorías
    final_selection = []
    for var in selected_weighted:
        add_var = True
        for kept_var in final_selection:
            if abs(df[var].corr(df[kept_var])) > umbral_colinealidad:
                add_var = False
                break
        if add_var:
            final_selection.append(var)

    # Asegurar que al menos una variable por categoría sea incluida si se solicita
    if conservar_minimo_por_categoria:
        for cat_name, cat_vars in variables_por_categoria.items():
            if cat_vars and not any(var in final_selection for var in cat_vars):
                # Encontrar la mejor variable de esta categoría
                best_var = max(cat_vars, key=lambda x: weighted_corr[x])
                final_selection.append(best_var)

    # 4. (Opcional) Ordenar por importancia de modelo
    if usar_modelo and final_selection:
        X = df[final_selection].fillna(method='ffill').fillna(method='bfill')
        y = df[target].fillna(method='ffill').fillna(method='bfill')

        # Verificar datos
        if len(X) > 0 and len(y) > 0:
            try:
                model = LGBMRegressor(random_state=random_state, verbose=-1)
                model.fit(X, y)
                importances = model.feature_importances_
                final_selection = [feat for _, feat in sorted(zip(importances, final_selection), reverse=True)]

                # Actualizar seleccionados por categoría
                selected_by_category = {cat: [] for cat in categorias.keys()}
                for var in final_selection:
                    for cat_name, suffix in categorias.items():
                        if suffix in var:
                            selected_by_category[cat_name].append(var)
                            break
            except Exception as e:
                print(f"Error en el entrenamiento del modelo: {e}")

    return final_selection, selected_by_category


# Función para visualizar resultados de la selección
def visualizar_seleccion(selected_vars: List[str], selected_by_category: Dict[str, List[str]]) -> None:
    """
    Visualiza las variables seleccionadas, organizadas por categoría.

    Args:
        selected_vars (List[str]): Lista de variables seleccionadas
        selected_by_category (Dict[str, List[str]]): Diccionario con variables por categoría
    """
    print(f"Total de variables seleccionadas: {len(selected_vars)}\n")

    for cat_name, vars_list in selected_by_category.items():
        if vars_list:
            print(f"{cat_name} ({len(vars_list)}):")
            for var in vars_list:
                print(f"  - {var}")
            print()