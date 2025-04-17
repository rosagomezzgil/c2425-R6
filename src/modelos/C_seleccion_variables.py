from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def seleccionar_variables(X, y, tecnica, umbral_correlacion=0.1):
    """
    Selecciona variables en función de la técnica especificada.
    Para 'var', se seleccionan aquellas con alta correlación con la variable objetivo.

    :param X: DataFrame de características
    :param y: Serie objetivo
    :param tecnica: Nombre del modelo (e.g. 'svr', 'var', etc.)
    :param umbral_correlacion: Mínimo valor absoluto de correlación para incluir una variable
    :return: X reducido y el selector (o None)
    """
    if tecnica == "svr":

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"var_{i}" for i in range(X.shape[1])])

        # SelectKBest con reducción real
        selector = SelectKBest(score_func=f_regression, k=min(15, X.shape[1]))
        X_kbest_array = selector.fit_transform(X, y)
        selected_columns = X.columns[selector.get_support()]
        X_kbest = pd.DataFrame(X_kbest_array, columns=selected_columns, index=X.index)

        # Reducir VIF iterativamente
        while True:
            vif = pd.Series(
                [variance_inflation_factor(X_kbest.values, i) for i in range(X_kbest.shape[1])],
                index=X_kbest.columns
            )
            max_vif = vif.max()
            if max_vif > 10:
                col_a_eliminar = vif.idxmax()
                X_kbest = X_kbest.drop(columns=[col_a_eliminar])
            else:
                break

        return X_kbest, selector

    elif tecnica == "var":
        # Concatenamos X con y para calcular correlaciones con la target
        df_completo = X.copy()
        df_completo["target"] = y
        correlaciones = df_completo.corr()["target"].drop("target")

        # Filtramos por correlación significativa
        variables_relevantes = correlaciones[correlaciones.abs() > umbral_correlacion].index.tolist()

        # Asegurarse de que haya al menos dos variables (VAR no funciona con una sola)
        if len(variables_relevantes) < 2:
            raise ValueError("No hay suficientes variables con correlación significativa para VAR.")

        X_filtrado = X[variables_relevantes]
        X_filtrado = X_filtrado.copy()
        X_filtrado.loc[:, "target"] = y

        return X_filtrado, None

    else:
        return X, None
