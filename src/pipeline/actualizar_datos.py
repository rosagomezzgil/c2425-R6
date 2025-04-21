import subprocess
import os
from datetime import datetime
import shutil
from src.utils.paths import get_path
import sys
from subprocess import run

""" 
Este script ejecuta de forma automatizada el flujo completo de actualización de datos, 
desde la extracción, pasando por la limpieza, hasta la exportación final de un archivo `.parquet`.

 Etapas del pipeline:
1. Extracción de datos desde múltiples fuentes externas (APIs, webs, etc.)
2. Limpieza y transformación de los datos extraídos.
3. Unión de todas las fuentes en un único DataFrame.
4. Eliminación de algunas variables y genereación de dataset_7d y dataset_30d
5. Creación de un backup de la versión anterior del conjunto final.
6. Guardado del nuevo archivo Parquet actualizado.
"""

def ejecutar_script(ruta):
    python_executable = sys.executable
    cwd = get_path("src")
    print(f"\n Ejecutando: {ruta}")
    resultado = subprocess.run([python_executable, ruta], cwd=cwd)
    if resultado.returncode != 0:
        print(f" Error al ejecutar {ruta}")
    else:
        print(f" Completado: {ruta}")

def ejecutar_notebook(ruta):
    python_executable = sys.executable
    cwd = get_path("src")
    print(f"\n Ejecutando notebook: {ruta}")

    resultado = run([
        python_executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.kernel_name=c2425-r6",
        ruta
    ], cwd=cwd)

    if resultado.returncode != 0:
        print(f" Error al ejecutar el notebook {ruta}")
    else:
        print(f" Notebook ejecutado correctamente: {ruta}")


if __name__ == "__main__":
    # -------- ETAPA 1: Extracción de datos --------
    print("\n ETAPA 1: Extracción de datos")
    rutas_extraccion = [
        get_path("src", "extraccion", "Alternative.py"),
        get_path("src", "extraccion", "Binance.py"),
        get_path("src", "extraccion", "Bitget.py"),
        get_path("src", "extraccion", "CoinGecko.py"),
        get_path("src", "extraccion", "Google_Trends.py"),
        get_path("src", "extraccion", "Halving.py"),
        get_path("src", "extraccion", "Hashrate.py"),
        get_path("src", "extraccion", "mercadoDerivados_binance.py"),
        get_path("src", "extraccion", "mercadoDerivados_deribit.py"),
        get_path("src", "extraccion", "Yahoo_Finance.py")
    ]
    for ruta in rutas_extraccion:
        ejecutar_script(ruta)

    # -------- ETAPA 2: Limpieza y transformación --------
    print("\n ETAPA 2: Limpieza y transformación")
    rutas_limpieza = [
        get_path("src", "transformacion", "limpieza_binance.py"),
        get_path("src", "transformacion", "limpieza_bitget.py"),
        get_path("src", "transformacion", "limpieza_coingecko.py"),
        get_path("src", "transformacion", "limpieza_datos_Google_Trends.py"),
        get_path("src", "transformacion", "limpieza_dxy.py"),
        get_path("src", "transformacion", "limpieza_fear_and_greed.py"),
        get_path("src", "transformacion", "limpieza_halving.py"),
        get_path("src", "transformacion", "limpieza_hashrate.py"),
        get_path("src", "transformacion", "limpieza_mercadoDerivados_binance.py"),
        get_path("src", "transformacion", "limpieza_mercDer_derivit.py"),
        get_path("src", "transformacion", "limpieza_Nasdaq.py"),
        get_path("src", "transformacion", "limpieza_sp500.py")
    ]
    for ruta in rutas_limpieza:
        ejecutar_script(ruta)

    # -------- ETAPA 3: Merge final y exportación --------
    print("\n ETAPA 3: Unión y exportación")
    ejecutar_script(get_path("src", "transformacion", "merge.py"))

    # -------- ETAPA 4: Generación de datasets de modelado --------
    print("\n ETAPA 4: Generación de datasets 7d y 30d")
    ejecutar_notebook(get_path("src", "modelos", "eliminacion de variables.ipynb"))

    # -------- ETAPA 5: BACKUP antes de guardar parquet final --------
    archivos_a_respaldar = {
        "data": "data.parquet",
        "data_7d": "data_7d.parquet",
        "data_30d": "data_30d.parquet",
    }

    backups_dir = get_path("data", "parquet", "backups")
    os.makedirs(backups_dir, exist_ok=True)

    for nombre_logico, nombre_archivo in archivos_a_respaldar.items():
        parquet_path = get_path("data", "parquet", nombre_archivo)

        if os.path.exists(parquet_path):
            fecha_hora = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_backup = f"{nombre_logico}_BACKUP_{fecha_hora}.parquet"
            ruta_backup = os.path.join(backups_dir, nombre_backup)
            shutil.copy2(parquet_path, ruta_backup)
            print(f" Backup creado para '{nombre_logico}': {ruta_backup}")
        else:
            print(f" No se encontró el archivo Parquet '{nombre_archivo}', no se creó backup.")

    # -------- ETAPA 6: Guardado del nuevo parquet -------
    ejecutar_script(get_path("src", "transformacion", "parquet_merged_data.py"))

    print("\n Pipeline de actualización completado.")
