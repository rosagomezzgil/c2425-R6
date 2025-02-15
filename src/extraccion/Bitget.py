import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Configurar Selenium con Chrome
options = Options()
options.add_argument("--headless")  # Modo sin interfaz gráfica
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Iniciar WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

# URL de Bitget (Datos históricos de Bitcoin)
url = "https://www.bitget.com/es/price/bitcoin/historical-data"
driver.get(url)

# Esperar a que la tabla cargue completamente
wait = WebDriverWait(driver, 10)
wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))

# Listas para almacenar datos
all_rows = []

while True:
    time.sleep(3)  # Esperar un poco para que cargue la página

    # **SOLUCIÓN: Volver a encontrar la tabla en cada iteración**
    try:
        table = driver.find_element(By.TAG_NAME, "table")
        rows = table.find_elements(By.TAG_NAME, "tr")
    except:
        print("Error al encontrar la tabla, reintentando...")
        continue  # Si hay error, vuelve a intentar

    # Obtener encabezados si es la primera página
    if not all_rows:
        headers = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]

    # Obtener todas las filas de la tabla
    page_data = []
    for row in rows[1:]:  # Omitir encabezados
        try:
            cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
            if cols:
                page_data.append(cols)
        except:
            print("Fila descartada por error de referencia.")
            continue

    # Agregar datos de la página actual
    all_rows.extend(page_data)

    # Buscar el botón "Next Page"
    try:
        next_button = driver.find_element(By.XPATH, "//li[contains(@class, 'ant-pagination-next')]")

        # Si está deshabilitado, salimos del bucle
        if "ant-pagination-disabled" in next_button.get_attribute("class"):
            print("Última página alcanzada. Finalizando...")
            break

        # Hacer clic en la flecha "Siguiente"
        driver.execute_script("arguments[0].click();", next_button)

        # Esperar que la nueva página cargue
        time.sleep(3)

    except Exception as e:
        print("No se encontró la flecha de 'Siguiente' o ya estamos en la última página.")
        break

# Cerrar el navegador
driver.quit()

# Convertir datos en DataFrame
df_bitget = pd.DataFrame(all_rows, columns=headers)

pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)  # Muestra todas las columnas
pd.set_option('display.expand_frame_repr', False)

# Guardar el conjunto de datos
output_path = r"../../data/bitget_scrapping.csv"
df_bitget.to_csv(output_path, encoding="utf-8-sig", index=False)

# Mostrar el DataFrame completo
print(df_bitget)

