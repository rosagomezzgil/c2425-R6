from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.base import BaseEstimator

def seleccionar_variables(X, y, tecnica, k_vars = 0):
    """
    Selecciona variables en función de la técnica especificada.
    Para 'var', se seleccionan aquellas con alta correlación con la variable objetivo.

    :param X: DataFrame de características
    :param y: Serie objetivo
    :param tecnica: Nombre del modelo (e.g. 'svr', 'var', etc.)
    :return: X reducido y el selector (o None)
    """
    if tecnica == "svr":
        resultados = {}

        # Guardar los nombres de las columnas originales
        original_columns = X.columns

        # 1. Usando f_regression (relación lineal)
        selector_f = SelectKBest(score_func=f_regression, k=min(k_vars, X.shape[1]))
        X_kbest_f = selector_f.fit_transform(X, y)
        selected_columns_indices_f = selector_f.get_support()
        selected_columns_f = original_columns[selected_columns_indices_f]
        resultados['f_regression'] = pd.DataFrame(X_kbest_f, columns=selected_columns_f, index=X.index)

        # 2. Usando mutual_info_regression (relación no lineal)
        selector_mi = SelectKBest(score_func=mutual_info_regression, k=min(k_vars, X.shape[1]))
        X_kbest_mi = selector_mi.fit_transform(X, y)
        selected_columns_indices_mi = selector_mi.get_support()
        selected_columns_mi = original_columns[selected_columns_indices_mi]
        resultados['mutual_info'] = pd.DataFrame(X_kbest_mi, columns=selected_columns_mi, index=X.index)

        # 3. Todas las variables
        resultados['all'] = X.copy()

        return resultados

    elif tecnica == "var":

        # 1. Comprobación de estacionariedad con ADF para cada variable
        # Test ADF solo sobre X (sin target)
        X_filtrado = X.copy()
        adf_fail = test_estacionariedad_adf(X_filtrado)

        # Diferenciación si hace falta
        if adf_fail:
            for col in adf_fail:
                X_filtrado.loc[:, col] = X_filtrado[col].diff()
            X_filtrado = X_filtrado.dropna()

        # 2. Aplicar PCA para reducir dimensionalidad
        # Estandarizamos las variables para PCA
        scaler = StandardScaler()
        X_estandarizado = scaler.fit_transform(X_filtrado)

        pca = PCA(n_components=0.90)  # Esto conserva el 90% de la varianza
        X_pca = pca.fit_transform(X_estandarizado)

        # Convertimos a DataFrame para compatibilidad
        X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i + 1}" for i in range(X_pca.shape[1])])

        # Retornamos el DataFrame con las componentes principales y la variable target
        X_pca_df["target"] = y

        return X_pca_df, None

    elif tecnica == "lstm":
        """
            Selecciona variables relevantes para un modelo LSTM usando pruebas de Granger Causality.

            :param X: DataFrame de características
            :param y: Serie objetivo
            :param tecnica: Técnica de modelado (para condicionales futuras)
            :param maxlag: Máximo desfase para la prueba de Granger
            :param threshold: Umbral de significancia para seleccionar variables
            :return: Subconjunto de X con variables seleccionadas y lista de seleccionadas
            """
        # Ajustar parámetros en función de mejores resultados
        maxlag = 5
        threshold = 0.05
        if tecnica != "lstm":
            raise NotImplementedError("Actualmente solo se implementa la selección para LSTM.")

        seleccionadas = []
        for col in X.columns:
            try:
                df_temp = pd.concat([y, X[col]], axis=1).dropna()
                df_temp.columns = ['target', 'feature']
                result = grangercausalitytests(df_temp, maxlag=maxlag, verbose=False)

                # Verifica si alguno de los lags tiene p-value < threshold
                p_values = [result[i + 1][0]['ssr_ftest'][1] for i in range(maxlag)]
                if any(p < threshold for p in p_values):
                    seleccionadas.append(col)
            except Exception as e:
                print(f"[Aviso] Error al procesar {col}: {e}")
                continue

        print(f"\n[Selección de Variables] Variables seleccionadas: {seleccionadas}")

        return X[seleccionadas], seleccionadas

    else:
        return X, None

def reducir_vif(X, umbral=10.0):
    """
    Elimina variables con VIF alto de forma iterativa.

    :param X: DataFrame con variables explicativas
    :param umbral: VIF máximo aceptable (por defecto: 10)
    :param verbose: Si True, imprime qué variables se eliminan
    :return: DataFrame con variables filtradas por VIF
    """
    X_filtrado = X.copy()
    while True:
        vif = pd.Series(
            [variance_inflation_factor(X_filtrado.values, i) for i in range(X_filtrado.shape[1])],
            index=X_filtrado.columns
        )
        max_vif = vif.max()
        if max_vif > umbral:
            col_a_eliminar = vif.idxmax()
            X_filtrado = X_filtrado.drop(columns=[col_a_eliminar])
        else:
            break
    return X_filtrado

def test_estacionariedad_adf(X, alpha=0.1):
    """
    Aplica el test de Dickey-Fuller aumentado (ADF) a cada columna de un DataFrame.

    Este test evalúa si una serie temporal es estacionaria. La hipótesis nula (H0) es que la serie
    NO es estacionaria (es decir, contiene raíz unitaria). Si el p-valor es mayor que el umbral alpha,
    no podemos rechazar la hipótesis nula, por lo que consideramos que esa variable no es estacionaria.

    :param X: DataFrame con variables temporales (una por columna)
    :param alpha: Nivel de significancia para el test ADF (por defecto 0.05)
    :return: Lista con los nombres de las columnas que NO son estacionarias
    """
    no_estacionarias = []
    for col in X.columns:
        p_valor = adfuller(X[col])[1]
        if p_valor > alpha:
            no_estacionarias.append(col)
    return no_estacionarias
