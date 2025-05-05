# CryptoVision Insights
![LOGO](https://img.bitgetimg.com/multiLang/web/ebb606b3d63bb1f7d48e7ccdfbb984c1.png)

# Presentación del Proyecto
El proyecto consiste en un sistema de **predicción de tendencias de Bitcoin**, que permite analizar los movimientos del mercado y prever posibles fluctuaciones en el precio de la criptomoneda. Para ello, se utilizan datos en tiempo real obtenidos mediante **web scraping** con Selenium y ChromeDriver, junto con modelos de análisis de datos.

El objetivo principal es proporcionar una herramienta que ayude a los inversores a tomar decisiones informadas basadas en patrones históricos y tendencias actuales del mercado.

# Estructura del Proyecto

```
CryptoVision-Insights/
├── src/                          # Código fuente principal organizado por etapas del flujo de datos
│   ├── 1 extraccion/            # Scripts para extraer datos desde múltiples fuentes
│   │   ├── Twitter/             # Extracción específica desde la API de Twitter
│   │   │   ├── config.py        # Configuración y autenticación para Twitter API
│   │   │   └── Twitter.py       # Script de conexión y extracción de tweets
│   │   ├── Alternative.py       # Fuente alternativa de métricas de mercado
│   │   ├── Binance.py           # Datos históricos de precios de Binance
│   │   ├── Bitget.py            # Datos de exchange Bitget
│   │   ├── CoinGecko.py         # API pública de criptomonedas
│   │   ├── Google_Trends.py     # Tendencias de búsqueda sobre cripto
│   │   ├── Halving.py           # Información sobre eventos de halving de Bitcoin
│   │   ├── Hashrate.py          # Métricas de poder de minado
│   │   ├── mercadoDerivados_binance.py  # Datos de derivados desde Binance
│   │   ├── mercadoDerivados_deribit.py  # Datos de derivados desde Deribit
│   │   └── Yahoo_Finance.py     # Datos financieros tradicionales (Nasdaq, SP500...)
│
│   ├── 2 transformacion/        # Procesamiento y limpieza de los datos crudos
│   │   ├── creacion_variables.py                # Creación de nuevas variables derivadas
│   │   ├── limpieza_*.py                        # Scripts de limpieza por fuente de datos
│   │   ├── merge.py                             # Fusión de todas las fuentes en un dataset único
│   │   ├── parquet_merged_data.py               # Exportación a formato parquet
│   │   └── sentiment_analysis.py                # Análisis de sentimiento de los tweets
│
│   ├── 3 analisis/              # Análisis exploratorio y visualización
│   │   ├── visualizacion merged datos.ipynb     # Visualización de datos ya fusionados
│   │   ├── visualización creacion variables.ipynb  # Análisis de variables generadas
│   │   └── visualización variable respuesta.ipynb  # Análisis de la variable objetivo
│
│   ├── 4 modelos/               # Modelado predictivo con diferentes algoritmos
│   │   ├── 7_dias/              # Modelos con ventana de predicción de 7 días
│   │   │   ├── ARBOLES.py       # Modelos tipo árbol
│   │   │   ├── LSTM.py          # Red neuronal LSTM
│   │   │   └── SVR.py           # Support Vector Regression
│   │   ├── 30_dias/             # Modelos con ventana de 30 días
│   │   │   ├── ARIMA.py
│   │   │   ├── SARIMA.py
│   │   │   └── XGBoost.py
│   │   ├── baseline.py          # Comparación con modelos base
│   │   ├── eliminación de variables.ipynb  # Feature selection
│   │   └── estudio de correlaciones.ipynb  # Análisis de correlación entre variables
│
│   ├── 5 pipeline/              # Automatización de tareas: entrenamiento y predicción
│   │   ├── modelos/             # Modelos entrenados y sus escaladores
│   │   │   ├── SVR.joblib
│   │   │   ├── SVR_scaler.joblib
│   │   │   ├── XGB.joblib
│   │   │   └── XGB_scaler.joblib
│   │   ├── actualizar_datos.py  # Script para actualizar predicciones automáticamente
│   │   └── reentrenamiento.py   # Reentrenamiento automático del modelo
│
│   └── 6 nuevos datos/          # Preparación de nuevos datos para predicciones en tiempo real
│       ├── creacion_nuevos_datos.py
│       ├── eliminación de variables nuevos datos.ipynb
│       └── merge_nuevos_datos.py
│
├── utils/                       # Funciones utilitarias y herramientas de ML
│   ├── arquitecturas/
│   │   ├── __init__.py
│   │   └── lstm_model.py        # Definición de arquitectura LSTM personalizada
│   ├── modelos_sklearn/         # Pipeline modular para modelos sklearn
│   │   ├── __init__.py
│   │   ├── A_cargar_datos.py             # Carga de datos
│   │   ├── B_preprocesamiento.py         # Preprocesamiento general
│   │   ├── C_seleccion_variables.py      # Selección de variables
│   │   ├── D_division_temporal.py        # División de datos en sets
│   │   ├── E_ajuste_hiperparametros.py   # Optimización de modelos
│   │   ├── F_metricas.py                 # Métricas de evaluación
│   │   └── G_graficar_predicciones.py    # Visualización de predicciones
│   ├── mlflow_utils.py          # Integración con MLflow para tracking
│   └── paths.py                 # Rutas comunes para acceso a datos y modelos
│
├── Dockerfile                   # Dockerfile para contenerizar la aplicación
├── requirements.txt             # Lista de dependencias (pip)
├── pyproject.toml               # Configuración del proyecto para herramientas modernas
├── uv.lock                      # Archivo de bloqueo de versiones con uv
├── .gitignore                   # Archivos ignorados por Git
└── README.md                    # Documentación del proyecto (este archivo)

# Indicaciones
El proyecto está desarrollado en **Python 3.10**, utilizando Selenium para la automatización de la extracción de datos. Para ejecutar o modificar el código, se recomienda utilizar **Jupyter Notebook** o un entorno de desarrollo como **VS Code** o **PyCharm**.
La carpeta de datos del [drive](https://drive.google.com/drive/folders/1rx4XvYLFwyn86TofruAenmslBwZ_eMg2?usp=sharing).

Para la correcta ejecución del proyecto, es necesario instalar las siguientes dependencias:

1. **Google Chrome**: Navegador web necesario para la automatización.
   - Instalación: [https://www.google.com/chrome/](https://www.google.com/chrome/)
2. **ChromeDriver**: Controlador que permite a Selenium interactuar con Chrome.
   - Instalación: [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)
3. **Instalación del Kernel de Jupyter en el entorno virtual**:  
   Para el pipeline correctamente asegurando la ejecución de los notebooks ejecutar el siguiente comando en la terminal con el entorno virtual activado
   ```bash
   python -m ipykernel install --user --name=c2425-r6 --display-name "Python (c2425-R6)"
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

