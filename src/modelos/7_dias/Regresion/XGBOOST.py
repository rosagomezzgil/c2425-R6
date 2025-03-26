import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Función para cargar datos desde un archivo Parquet
def cargar_datos(ruta_parquet):
    df = pd.read_parquet(ruta_parquet)
    return df


# Función para preprocesar los datos
def preprocesar_datos(df):
    df = df.drop(columns=['tendencia_futura_7d'], errors='ignore')
    X = df.drop(columns=['pct_cambio_7d'])
    y = df['pct_cambio_7d']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# Función para dividir los datos en train, val y test, de manera ordenada en el tiempo, siendo
# el conjunto de entrenamiento el 80% de los datos más antiguos, validación el siguiente 10% y prueba el último 10%
def dividir_datos(X, y, test_size=0.1, val_size=0.1):
    n = len(X)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


# Función para buscar los mejores hiperparámetros para SVR
def buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val):
    parametros = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    grid_search = GridSearchCV(xgb_model, parametros, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    mejor_modelo = grid_search.best_estimator_
    mejores_parametros = grid_search.best_params_

    # Evaluamos el mejor modelo sobre el conjunto de validación
    val_preds = mejor_modelo.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_preds)
    val_rmse = mean_squared_error(y_val, val_preds, squared=False)

    print(f"Mejores Hiperparámetros: {mejores_parametros}")
    print(f"Evaluación sobre validación - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")

    #print("Mejores Hiperparámetros:", mejores_parametros)

    return mejor_modelo, mejores_parametros

# Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")


# Función principal
def main():
    ruta_parquet = r"../../../../data/parquet/data_7d.parquet"
    df = cargar_datos(ruta_parquet)

    X, y, scaler = preprocesar_datos(df)

    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)

    mejor_modelo, mejores_parametros = buscar_mejores_hiperparametros(X_train, y_train, X_val, y_val)

    evaluar_modelo(mejor_modelo, X_test, y_test)

if __name__ == "__main__":
    main()
