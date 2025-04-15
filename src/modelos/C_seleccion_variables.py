from sklearn.feature_selection import SelectKBest, f_regression

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
        selector = SelectKBest(score_func=f_regression, k="all")
        X_new = selector.fit_transform(X, y)
        return X_new, selector

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
