import pandas as pd

# Datos de Halving
def get_halving_dates():
    halvings = [
        {"Fecha": "2012-11-28", "Bloque": 210000, "Recompensa": 25},
        {"Fecha": "2016-07-09", "Bloque": 420000, "Recompensa": 12.5},
        {"Fecha": "2020-05-11", "Bloque": 630000, "Recompensa": 6.25},
        {"Fecha": "2024-04-XX", "Bloque": 840000, "Recompensa": 3.125},  # Estimado
    ]
    df_halving = pd.DataFrame(halvings)
    df_halving["Fecha"] = pd.to_datetime(df_halving["Fecha"], errors="coerce")
    return df_halving


if __name__ == "__main__":
    # Obtener datos de halving
    df_halving = get_halving_dates()

    output_path = r"../../data/extraccion/halving_data.csv"
    df_halving.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df_halving)
