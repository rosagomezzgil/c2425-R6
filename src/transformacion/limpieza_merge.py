import pandas as pd
import numpy as np

def calcular_retorno_logaritmico(series: pd.Series):
    """
    Calcula el retorno logarítmico de una serie temporal de precios.

    :param series: Serie de precios en el tiempo.
    :return: Serie de retornos logarítmicos.
    """
    return np.log(series / series.shift(1))

def evaluar(column: str, data_source_1: str, data_source_2: str, df: pd.DataFrame) -> tuple[float, float]:
    """
    Evalúa la similitud entre dos fuentes de datos mediante los retornos logarítmicos.

    :param column: Nombre base de la columna (sin sufijo de fuente de datos).
    :param data_source_1: Primera fuente de datos (ejemplo: 'binance').
    :param data_source_2: Segunda fuente de datos (ejemplo: 'bitget' o 'coingecko').
    :param df: DataFrame con los datos.
    :return: Una tupla con el Error Absoluto Medio (MAE) y la correlación de Pearson de los retornos.
    """

    column_1, column_2 = f"{column}_{data_source_1}", f"{column}_{data_source_2}"

    # Verificar que las columnas existen en el DataFrame
    assert column_2 in df, f"Columna no encontrada: {column_2}"
    assert column_1 in df, f"Columna no encontrada: {column_1}"

    # Eliminar filas con valores nulos
    df_filtered = df[[column_1, column_2]].dropna()

    # Calcular retornos logarítmicos
    df_filtered = df_filtered[(df_filtered[column_1] > 0) & (df_filtered[column_2] > 0)]
    df_filtered[f"{column_1}_retornos"] = calcular_retorno_logaritmico(df_filtered[column_1])
    df_filtered[f"{column_2}_retornos"] = calcular_retorno_logaritmico(df_filtered[column_2])

    # Eliminar NaNs generados por el cálculo de retornos
    df_filtered = df_filtered.dropna()

    # Calcular MAE y correlación de Pearson en los retornos
    mae = np.mean(np.abs(df_filtered[f"{column_1}_retornos"] - df_filtered[f"{column_2}_retornos"]))
    pearson = np.corrcoef(df_filtered[f"{column_1}_retornos"], df_filtered[f"{column_2}_retornos"])[0, 1]

    print(f"MAE entre {data_source_1.capitalize()} y {data_source_2.capitalize()} ({column}): {mae:.5f}")
    print(f"Correlación Pearson (r) entre {data_source_1.capitalize()} y {data_source_2.capitalize()} ({column}): {pearson:.5f}")

    return mae, pearson

def interpretar_metrica(mae: float, r: float, nombre: str):
    """
    Evalúa la relación entre los datos según el error absoluto medio (MAE)
    y la correlación de Pearson (r), proporcionando una conclusión interpretativa.

    :param mae: Error Absoluto Medio (MAE) entre ambas fuentes.
    :param r: Coeficiente de correlación de Pearson.
    :param nombre: Nombre de la métrica evaluada.
    :return: Una descripción interpretativa de la coherencia entre los datos.
    """
    if r > 0.95:
        if mae > 1000:
            return f"{nombre}: Diferencia alta en valores absolutos, pero sigue la misma tendencia. No es un problema grave."
        else:
            return f"{nombre}: Diferencia baja y sigue la misma tendencia. Métricas consistentes."
    elif r > 0.7:
        return f"{nombre}: Buena relación, pero hay diferencias moderadas en valores."
    elif r > 0.3:
        return f"{nombre}: Relación débil. No coinciden bien en esta métrica."
    elif r > 0:
        return f"{nombre}: Casi sin relación. Tienen diferencias importantes aquí."
    else:
        return f"{nombre}: Relación negativa. Parecen moverse en direcciones opuestas. Algo está mal."

if __name__ == "__main__":
    # Cargar los datos
    file = r'../../data/limpieza/merged_data.csv'
    merged = pd.read_csv(file, header=0)

    # Convertir timestamp a tipo datetime
    merged['timestamp'] = pd.to_datetime(merged['timestamp'])

    # Filtrar solo los últimos 3 años
    fecha_mas_actual = merged['timestamp'].max()
    fecha_limite = fecha_mas_actual - pd.DateOffset(years=3)
    merged_limpio = merged[merged['timestamp'] >= fecha_limite]

    """
        1. Evaluar Binance comparándolo con Bitget
    """
    # Columnas a evaluar
    columns = [
        'open', 'high', 'low', 'close', 'volumen',
        'var_diaria', 'volatilidad_50', 'volatilidad_200', 'rango_dia',
        'momentum_50', 'momentum_200', 'SMA_50', 'SMA_200',
        'EMA_50', 'EMA_200', 'RSI'
    ]

    # Calcular métricas para cada columna
    resultados = [evaluar(column, 'binance', 'bitget', merged_limpio) for column in columns]

    print('\nInterpretación de las métricas:')
    for column, mae, r in zip(columns, *zip(*resultados)):
        print(interpretar_metrica(mae, r, column))
    print()

    """
    En general, Binance y Bitget muestran una alta coherencia en la mayoría de las métricas clave,
    con diferencias mínimas en precios, volatilidad y medias móviles,
    lo que indica que ambas plataformas reflejan tendencias similares del mercado.
    """

    """
            2. Evaluar Binance comparándolo con Coingecko
    """
    # Columnas a evaluar
    columns = [
        'close', 'volumen',
        'var_diaria', 'volatilidad_50', 'volatilidad_200', 'rango_dia',
        'momentum_50', 'momentum_200', 'SMA_50', 'SMA_200',
        'EMA_50', 'EMA_200', 'RSI'
    ]

    # Calcular métricas para cada columna
    resultados = [evaluar(column, 'binance', 'coingecko', merged_limpio) for column in columns]

    print('\nInterpretación de las métricas:')
    for column, mae, r in zip(columns, *zip(*resultados)):
        print(interpretar_metrica(mae, r, column))
    print()

    """
    Los datos muestran que Binance y Coingecko presentan discrepancias significativas en varias métricas clave, 
    especialmente en el precio de cierre, volumen, variación diaria y momentum,
    donde las correlaciones son negativas, indicando que los valores de ambas plataformas no siguen la misma tendencia.
    """

    """
        3. Evaluar los mercados de derivados
    """

    # Columnas a evaluar
    columns = [
        'mDopen', 'mDhigh', 'mDlow', 'mDclose', 'mDvolume',
        'mDvar_diaria', 'mDvolatilidad_50', 'mDvolatilidad_200', 'mDrango_dia',
        'mDmomentum_50', 'mDmomentum_200', 'mDSMA_50', 'mDSMA_200',
        'mDEMA_50', 'mDEMA_200', 'mD_RSI'
    ]

    # Calcular métricas para cada columna
    resultados = [evaluar(column, 'mDbinance', 'mDderibit', merged_limpio) for column in columns]

    print('\nInterpretación de las métricas:')
    for column, mae, r in zip(columns, *zip(*resultados)):
        print(interpretar_metrica(mae, r, column))
    print()

    """
    Las plataformas Binance y Deribit muestran una alta concordancia en los datos,
    especialmente en términos de precios y métricas clave. Las diferencias observadas,
    como en volumen y algunas métricas de largo plazo (como EMA 200 y rango diario),
    aunque notables, son moderadas en comparación con la similitud general.
    Estas métricas son lo suficientemente consistentes como para confiar en que ambas plataformas 
    siguen la misma tendencia de mercado
    """

    """
        4. Dejar únicamente las columnas de Binance y hashrate, google trends
    """

    eliminar = ['bitget', 'coingecko', 'mDderibit']

    # Filtramos las columnas que no contienen ninguna de las palabras clave
    merged_limpio = merged_limpio.loc[:, ~merged_limpio.columns.str.contains('|'.join(eliminar))]

    # Guardar el dataframe final
    output_path = r'../../data/limpieza/cleaned_data.csv'
    merged_limpio.to_csv(output_path, index=False)

