# CryptoVision Insights
![LOGO](https://img.bitgetimg.com/multiLang/web/ebb606b3d63bb1f7d48e7ccdfbb984c1.png)

# Presentación del Proyecto
El proyecto consiste en un sistema de **predicción de tendencias de Bitcoin**, que permite analizar los movimientos del mercado y prever posibles fluctuaciones en el precio de la criptomoneda. Para ello, se utilizan datos en tiempo real obtenidos mediante **web scraping** con Selenium y ChromeDriver, junto con modelos de análisis de datos.

El objetivo principal es proporcionar una herramienta que ayude a los inversores a tomar decisiones informadas basadas en patrones históricos y tendencias actuales del mercado.

# Indicaciones
El proyecto está desarrollado en **Python 3.10**, utilizando Selenium para la automatización de la extracción de datos. Para ejecutar o modificar el código, se recomienda utilizar **Jupyter Notebook** o un entorno de desarrollo como **VS Code** o **PyCharm**.
La carpeta de datos del [drive](https://drive.google.com/drive/folders/1rx4XvYLFwyn86TofruAenmslBwZ_eMg2?usp=sharing).

Para la correcta ejecución del proyecto, es necesario instalar las siguientes dependencias:

1. **Google Chrome**: Navegador web necesario para la automatización.
   - Instalación: [https://www.google.com/chrome/](https://www.google.com/chrome/)
2. **ChromeDriver**: Controlador que permite a Selenium interactuar con Chrome.
   - Instalación: [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)
3. **CUDA 11.2**: plataforma de computación paralela de NVIDIA que permite ejecutar cálculos en la GPU
   - Instalación: [https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
4. **cuDNN 8.1.0**: biblioteca de NVIDIA optimizada para redes neuronales
   - Instalación: [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)
    **Extrae el ZIP** y copia estos archivos dentro del directorio de instalación de CUDA (normalmente en `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`):

     - Copia el contenido de `bin\` a `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
     - Copia el contenido de `lib\` a `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`
     - Copia el contenido de `include\` a `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`


# Estructura del Proyecto

```
src/: Contiene el código fuente del proyecto.

extraccion/: Administra la obtención de datos desde diferentes fuentes.
    Twitter/
        config.py: Configuración general para la extracción de datos.
        Twitter.py: Obtiene datos desde Twitter relacionados con Bitcoin.
        Alternative.py, Binance.py, Bitget.py: Scripts para obtener datos desde distintas plataformas de intercambio.
        CoinGecko.py: Obtiene datos desde la API de CoinGecko.
        Google_Trends.py: Extrae datos de tendencias de Google.
        Yahoo_Finance.py: Obtiene información financiera desde Yahoo Finance.
        Otros archivos: Scripts adicionales para extraer datos relevantes.

transformacion/: Gestiona el procesamiento y limpieza de los datos obtenidos.
    CleanBitcoinCoingecko.py: Limpia y estructura datos de CoinGecko.
    Creacion_datos_binance.py: Genera un dataset estructurado con datos de Binance.
    limpieza_binance.py, limpieza_bitget.py: Scripts de limpieza específicos para plataformas de intercambio.
    limpieza_datos_Google_Trends.py: Procesa y limpia datos de Google Trends.
    limpieza_Nasdaq.py: Filtra y organiza datos de Nasdaq.
    Sentiment_Analysis.py: Realiza análisis de sentimiento basado en los datos extraídos.

.gitignore: Especifica los archivos y directorios que Git debe ignorar.

requirements.txt: Lista las dependencias necesarias para el proyecto.

README.md: Contiene la información general del proyecto.
```

# Reglas del Análisis

### 1. Obtención de Datos
- Se extraen datos de precios de Bitcoin desde plataformas como **CoinMarketCap**, **Binance**, **Bitget** y **Yahoo Finance**.
- Se almacena la información en formatos como CSV o bases de datos para su posterior análisis.

### 2. Procesamiento de Datos
- Se filtran y limpian los datos recopilados.
- Se generan estadísticas descriptivas y se identifican patrones históricos.

### 3. Predicción de Tendencias
- Se utilizan modelos de machine learning como **ARIMA, LSTM** o **Redes Neuronales** para prever tendencias futuras.
- Se comparan los resultados de diferentes modelos para optimizar la predicción.


# Crédito
Rodrigo Jesús-Portanet    
Pablo Alonso    
Rosa Gómez-Gil    
Ignacio Ramírez    
Daniel Higueras    
Vega García   

