import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv(r"../../data/datos_google_trends.csv")
    df = df.reset_index(drop=True)

    # Convertir timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Crear un rango de fechas completo con frecuencia diaria
    df_complete = pd.DataFrame(
        {'timestamp': pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')})

    # Unir con el dataset original
    df_merged = pd.merge(df_complete, df, on='timestamp', how='left')

    #Estudiar el numero de valores faltantes
    valores_faltantes = df.isna().sum()
    print(f'Hay {valores_faltantes} valores faltantes en total')

    #Remplazar los valores faltantes por 0 ya que significa que no ha habido busquedas ese dia sobre el bitcoin
    df_merged.fillna(0, inplace=True)

    # Guardar el conjunto de datos
    output_path = r"../../data/limpios_google_trends.csv"
    df_merged.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df_merged.head())
    print(df_merged.info())
