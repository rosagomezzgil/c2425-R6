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


# Estructura del Proyecto

```
src/
│
├── analisis/                     # Notebooks de análisis y visualización de datos
│   ├── Visualizacion_merged_datos.ipynb
│   ├── Visualización_creacion_variables.ipynb
│   └── Visualización_variable_respuesta.ipynb
│
├── extraccion/                  # Scripts para extracción de datos desde diversas fuentes
│   ├── Twitter/
│   │   ├── config.py            # Configuración de API para Twitter
│   │   └── Twitter.py           # Extracción de datos desde Twitter
│   ├── Alternative.py           # Datos de la plataforma Alternative.me (índice de miedo/codicia, etc.)
│   ├── Binance.py               # Extracción de datos del exchange Binance
│   ├── Bitget.py                # Extracción de datos del exchange Bitget
│   ├── CoinGecko.py             # API pública de CoinGecko
│   ├── Google_Trends.py         # Tendencias de búsqueda en Google
│   ├── Halving.py               # Info sobre eventos de halving de Bitcoin
│   ├── Hashrate.py              # Datos de hashrate de Bitcoin
│   ├── mercadoDerivados_binance.py
│   ├── mercadoDerivados_deribit.py
│   └── Yahoo_Finance.py         # Datos financieros generales desde Yahoo Finance
│
├── modelos/                     # Entrenamiento y predicción de modelos ML/DL
│   ├── 7_dias/                  # Modelos para predicciones a 7 días
│   │   ├── ARBOLES_DE_DECISION.py
│   │   ├── LSTM.py
│   │   └── SVR.py
│   ├── 30_dias/                 # Modelos para predicciones a 30 días
│   │   ├── ARIMA.py
│   │   ├── SARIMA.py
│   │   └── VAR.py
│   ├── A_cargar_datos.py
│   ├── B_preprocesamiento.py
│   ├── C_seleccion_variables.py
│   ├── D_division_temporal.py
│   ├── E_ajuste_hiperparametros.py
│   ├── F_metricas.py
│   ├── G_graficar_predicciones.py
│   ├── baseline.py
│   ├── eliminacion de variables.ipynb
│   └── Estudio_correlaciones.ipynb
│
├── transformacion/             # Limpieza, procesamiento y transformación de datos
│   ├── Calcular_RSI.py
│   ├── limpieza_binance.py
│   ├── limpieza_bitget.py
│   ├── limpieza_coingecko.py
│   ├── limpieza_datos_Google_Trends.py
│   ├── limpieza_dxy.py
│   ├── limpieza_fear_and_greed.py
│   ├── limpieza_halving.py
│   ├── limpieza_hashrate.py
│   ├── limpieza_mercadoDerivados_binance.py
│   ├── limpieza_mercDer_derivit.py
│   ├── limpieza_Nasdaq.py
│   ├── limpieza_sp500.py
│   ├── merge.py
│   ├── parquet_merged_data.py
│   └── Sentiment Analysis.py # Procesamiento de sentimiento en noticias
│
├── .gitignore                   # Archivos/Directorios ignorados por Git
├── pyproject.toml               # Configuración del entorno y dependencias
├── uv.lock                      # Archivo de bloqueo de dependencias para `uv` (gestor)
└── README.md                    # Documentación general del proyecto

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

