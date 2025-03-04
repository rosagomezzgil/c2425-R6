import pandas as pd
import numpy as np

def evaluar(column: str, data_source_1: str, data_source_2: str, df: pd.DataFrame):
    """
    Calcula el Error Absoluto Medio (MAE) y la correlación de Pearson entre dos fuentes de datos (Binance/Bitget o Binance/Coingecko).
    :param column: Nombre base de la columna (sin el sufijo _binance o _bitget).
    :param data_source_1: Primer fuente de datos (por ejemplo, 'binance').
    :param data_source_2: Segunda fuente de datos (por ejemplo, 'bitget' o 'coingecko').
    :param df: DataFrame con los datos.
    :return: (MAE, Pearson)
    """

    column_1, column_2 = column + '_' + data_source_1, column + '_' + data_source_2

    assert column_2 in df, f"Columna no encontrada: {column_2}"
    assert column_1 in df, f"Columna no encontrada: {column_1}"

    # Eliminar filas con valores nulos en cualquiera de las dos series
    df_filtered = df[[column_1, column_2]].dropna()

    # Extraer los valores limpios
    source_1_values = df_filtered[column_1].values
    source_2_values = df_filtered[column_2].values

    # Calcular MAE
    mae = np.mean(np.abs(source_1_values - source_2_values))

    # Calcular Pearson (si hay variabilidad en los datos)
    if np.std(source_1_values) == 0 or np.std(source_2_values) == 0:
        pearson = np.nan  # No se puede calcular correlación si hay varianza cero
    else:
        pearson = np.corrcoef(source_1_values, source_2_values)[0, 1]

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
    file = r'../../data/merged_data.csv'
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
    En términos de precios y volatilidad, Binance y Bitget se comportan de manera casi idéntica, siguiendo las mismas tendencias, aunque con algunas diferencias en magnitud.
    Las métricas como varianza diaria, volatilidad y rango del día son consistentes entre ambas plataformas, lo que sugiere que ambas tienen patrones similares de movimiento en el mercado.
    El volumen y las medias móviles también muestran una alta similitud en las tendencias, aunque con diferencias en magnitudes.
    La única excepción notable es el RSI, que muestra una correlación negativa, lo que indica que Binance y Bitget pueden tener diferentes señales sobre la fuerza o debilidad del mercado en un mismo período
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
    En términos generales, Binance y Coingecko muestran una tendencia similar en muchas métricas clave,
    como el precio de cierre, volatilidad y momentum.
    La relación entre estas dos plataformas es fuerte en la mayoría de las métricas,
    lo que indica que los movimientos de precios en ambas plataformas son en gran medida consistentes.
    
    Volumen muestra una relación débil, lo que sugiere que los datos de volumen entre ambas plataformas no coinciden bien.
    La varianza diaria y el rango del día presentan discrepancias significativas,
    con una correlación negativa en la varianza y una baja correlación en el rango del día,
    lo que podría indicar diferencias en la forma en que cada plataforma maneja estos datos.
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
    En términos generales, las plataformas Binance y Deribit tienen una correlación muy alta en casi todas las métricas clave,
    lo que sugiere que las tendencias de los precios y otros indicadores siguen patrones muy similares en ambas plataformas.
    Sin embargo, algunas diferencias en volumen y rango del día sugieren
    que pueden haber variaciones en la forma en que cada plataforma maneja o reporta ciertos datos.
    A pesar de estas diferencias, las métricas muestran una alta consistencia en su comportamiento general,
    lo que implica que las dos plataformas están alineadas en la mayoría de los aspectos de análisis técnico y de mercado.
    """

    """
        4. Dejar únicamente las columnas de Binance y hashrate, google trends
    """

    eliminar = ['bitget', 'coingecko', 'mDderibit']

    # Filtramos las columnas que no contienen ninguna de las palabras clave
    merged_limpio = merged_limpio.loc[:, ~merged_limpio.columns.str.contains('|'.join(eliminar))]

    # Guardar el dataframe final
    output_path = r'../../data/limpio_merged_data.csv'
    merged_limpio.to_csv(output_path, index=False)

