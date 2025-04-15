from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
import numpy as np

def ajustar_modelo(modelo, X, y, param_grid, strategy="grid", scoring="neg_mean_squared_error", n_splits=5):
    """
    Ajusta un modelo usando validación cruzada temporal con GridSearch o RandomSearch.

    :param modelo: Modelo sklearn
    :param X: Características
    :param y: Objetivo
    :param param_grid: Hiperparámetros a buscar
    :param strategy: 'grid' o 'random'
    :param scoring: Métrica de evaluación
    :param n_splits: Número de divisiones temporales

    :return: Mejor estimador, mejores parámetros
    """
    cv = TimeSeriesSplit(n_splits=n_splits)

    if strategy == "grid":
        search = GridSearchCV(modelo, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    else:
        search = RandomizedSearchCV(modelo, param_grid, scoring=scoring, cv=cv, n_iter=20, n_jobs=-1)

    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def validar_modelo_timeseries(y, param_grid, model_class, seasonal=False, seasonal_param_grid=None, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_mse = np.inf
    best_params = None
    best_model = None

    if seasonal and seasonal_param_grid:
        for order in param_grid:
            for seasonal_order in seasonal_param_grid:
                fold_mse = []
                for train_idx, test_idx in tscv.split(y):
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    try:
                        model = model_class(y_train, order=order, seasonal_order=seasonal_order)
                        model_fit = model.fit(disp=False)
                        pred = model_fit.predict(start=y_test.index[0], end=y_test.index[-1])
                        mse = np.mean((y_test - pred[:len(y_test)]) ** 2)
                        fold_mse.append(mse)
                    except Exception:
                        fold_mse.append(np.inf)

                avg_mse = np.mean(fold_mse)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_params = (order, seasonal_order)
                    best_model = model_class(y, order=order, seasonal_order=seasonal_order).fit(disp=False)
    else:
        for order in param_grid:
            fold_mse = []
            for train_idx, test_idx in tscv.split(y):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                try:
                    model = model_class(y_train, order=order)
                    model_fit = model.fit()
                    pred = model_fit.predict(start=y_test.index[0], end=y_test.index[-1])
                    mse = np.mean((y_test - pred[:len(y_test)]) ** 2)
                    fold_mse.append(mse)
                except Exception:
                    fold_mse.append(np.inf)

            avg_mse = np.mean(fold_mse)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = order
                best_model = model_class(y, order=order).fit()

    return best_model, best_params


def validar_con_timeseriessplit_var(df, param_grid, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_mse = np.inf
    best_lag = None
    best_model = None

    for lag in param_grid:
        fold_mse = []

        for train_idx, test_idx in tscv.split(df):
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

            try:
                model = VAR(train_df)
                model_fit = model.fit(lag)
                pred = model_fit.forecast(train_df.values[-lag:], steps=len(test_df))
                mse = np.mean((test_df.values - pred) ** 2)
                fold_mse.append(mse)

            except Exception as e:
                print(f"Error en lag={lag}: {e}")
                fold_mse.append(np.inf)

        avg_mse = np.mean(fold_mse)
        print(f"Lag={lag}, avg MSE={avg_mse:.4f}")

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_lag = lag
            try:
                best_model = VAR(df).fit(lag)
            except Exception as e:
                print(f"Error final al ajustar modelo VAR con lag={lag}: {e}")
                best_model = None

    if best_model is None:
        print("⚠️ No se encontró ningún modelo válido. Revisa los datos o el rango de lags.")

    return best_model, best_lag
