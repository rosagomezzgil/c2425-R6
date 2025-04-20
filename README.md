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
├── analisis/                    # Notebooks para análisis exploratorio y visualización de variables
│   ├── Visualizacion_merged_datos.ipynb         # Análisis visual de datos ya fusionados
│   ├── Visualización_creacion_variables.ipynb   # Visualización de variables generadas/ingeniería de características
│   └── Visualización_variable_respuesta.ipynb   # Análisis visual de la variable objetivo
│
├── extraccion/                  # Scripts para la extracción de datos desde distintas APIs y fuentes externas
│   ├── Twitter/
│   │   ├── config.py            # Configuración de autenticación para la API de Twitter (X)
│   │   └── Twitter.py           # Script para recolectar datos desde Twitter
│   ├── Alternative.py           # Extracción del índice Fear & Greed desde Alternative.me
│   ├── Binance.py               # Descarga de datos históricos del exchange Binance
│   ├── Bitget.py                # Obtención de datos de mercado desde el exchange Bitget
│   ├── CoinGecko.py             # Acceso a datos de criptomonedas vía API pública de CoinGecko
│   ├── Google_Trends.py         # Consulta de tendencias de búsqueda relacionadas a criptomonedas en Google
│   ├── Halving.py               # Información histórica y futura sobre eventos de halving de Bitcoin
│   ├── Hashrate.py              # Datos sobre el hashrate de la red de Bitcoin
│   ├── mercadoDerivados_binance.py   # Datos de derivados financieros desde Binance
│   ├── mercadoDerivados_deribit.py   # Extracción de datos de derivados desde Deribit
│   └── Yahoo_Finance.py         # Recolección de datos macroeconómicos y financieros generales desde Yahoo Finance
│
├── modelos/                     # Entrenamiento, evaluación y comparación de modelos de predicción
│   ├── 7_dias/                  # Modelos entrenados para realizar predicciones a 7 días
│   │   ├── ARBOLES_DE_DECISION.py     # Modelo de árboles de decisión
│   │   ├── LSTM.py                     # Modelo LSTM con PyTorch
│   │   └── SVR.py                      # Máquina de soporte vectorial para regresión
│   ├── 30_dias/                 # Modelos estadísticos para predicciones a 30 días
│   │   ├── ARIMA.py                   # Modelo ARIMA
│   │   ├── SARIMA.py                  # Modelo SARIMA
│   │   └── VAR.py                     # Modelo VAR (vectores autorregresivos)
│   ├── arquitecturas/           # Arquitecturas personalizadas compatibles con los estimadores de Scikit-learn
│   │   └── lstm_model.py              # Clase LSTM como estimador compatible con Scikit-learn
│   ├── baseline.py              # Modelo de referencia (baseline) para comparación
│   ├── eliminacion de variables.ipynb # Estudio de importancia/eliminación de variables
│   └── Estudio_correlaciones.ipynb   # Análisis de correlación entre variables
│
├── transformacion/              # Procesamiento, limpieza y transformación de datos crudos
│   ├── Calcular_RSI.py                      # Cálculo del indicador técnico RSI
│   ├── limpieza_binance.py                 # Limpieza de datos de Binance
│   ├── limpieza_bitget.py                  # Limpieza de datos de Bitget
│   ├── limpieza_coingecko.py               # Limpieza de datos de CoinGecko
│   ├── limpieza_datos_Google_Trends.py     # Preprocesamiento de tendencias de búsqueda de Google
│   ├── limpieza_dxy.py                     # Limpieza de datos del índice DXY (dólar)
│   ├── limpieza_fear_and_greed.py          # Procesamiento del índice Fear & Greed
│   ├── limpieza_halving.py                 # Estandarización de datos sobre halving de BTC
│   ├── limpieza_hashrate.py                # Procesamiento de datos de hashrate
│   ├── limpieza_mercadoDerivados_binance.py# Limpieza de datos derivados de Binance
│   ├── limpieza_mercDer_derivit.py         # Limpieza de datos derivados de Deribit
│   ├── limpieza_Nasdaq.py                  # Limpieza de datos del índice Nasdaq
│   ├── limpieza_sp500.py                   # Limpieza de datos del índice S&P 500
│   ├── merge.py                            # Unión e integración de todas las fuentes de datos
│   ├── parquet_merged_data.py              # Generación y guardado de datos integrados en formato Parquet
│   └── Sentiment Analysis.py               # Análisis de sentimiento en noticias y tweets
│
├── utils/                       # Utilidades y componentes reutilizables para entrenamiento y evaluación
│   ├── modelos_sklearn/         # Pipeline completo para entrenamiento de modelos clásicos
│   │   ├── A_cargar_datos.py           # Lectura y carga de datasets
│   │   ├── B_preprocesamiento.py       # Preprocesamiento general (escalado, imputación, etc.)
│   │   ├── C_seleccion_variables.py    # Selección de variables relevantes
│   │   ├── D_division_temporal.py      # Separación train/test respetando la estructura temporal
│   │   ├── E_ajuste_hiperparametros.py # Ajuste de hiperparámetros con validación cruzada temporal
│   │   ├── F_metricas.py               # Cálculo de métricas de evaluación (RMSE, MAE, etc.)
│   │   └── G_graficar_predicciones.py  # Visualización de predicciones vs valores reales
│   ├── mlflow_utils.py                # Funciones de utilidad para registrar y gestionar modelos con MLflow
│   └── paths.py                       # Rutas centralizadas a carpetas y archivos del proyecto
│
├── .gitignore                   # Define qué archivos/directorios no se deben subir al repositorio
├── pyproject.toml               # Archivo de configuración del entorno y dependencias del proyecto
├── uv.lock                      # Bloqueo de versiones de dependencias usado por `uv`
└── README.md                    # Descripción general, propósito y estructura del proyecto
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

