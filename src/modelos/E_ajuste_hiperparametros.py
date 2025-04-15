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

