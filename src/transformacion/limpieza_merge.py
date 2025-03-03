import pandas as pd

if __name__ == "__main__":
    # Cargar los datos
    file = r'../../data/merged_data.csv'
    merged = pd.read_csv(file, header=0)

    # Convertir la columna timestamp a tipo datetime
    merged['timestamp'] = pd.to_datetime(merged['timestamp'])

    # Obtener la fecha más reciente en el dataset
    fecha_mas_actual = merged['timestamp'].max()

    # Restar 3 años
    fecha_limite = fecha_mas_actual - pd.DateOffset(years=3)

    # Filtrar solo los últimos 3 años
    merged_limpio = merged[merged['timestamp'] >= fecha_limite]

    """
        1. Estudio de nulos
    """

    print(merged_limpio.isnull().sum(axis=0))

    #Guardar el dataframe final
    output_path = r'../../data/limpio_merged_data.csv'
    merged_limpio.to_csv(output_path, index=False)
