# config.py
"""
Módulo: configuración de credenciales para autenticación con la API de Twitter.

Este script contiene las claves necesarias para autenticar el acceso a la API de Twitter
(v2) usando la librería `tweepy`. Es utilizado por los módulos de extracción de datos
que requieren conexión a dicha API.

IMPORTANTE:
Este archivo contiene información sensible (API keys y tokens). No debe ser compartido
ni subido a repositorios públicos. Se recomienda incluirlo en el archivo `.gitignore`
y considerar el uso de variables de entorno o gestores de secretos para ambientes productivos.
"""

API_KEY = "***REMOVED***"
API_SECRET = "***REMOVED***"
ACCESS_TOKEN = "***REMOVED***"
ACCESS_SECRET = "***REMOVED***"
BEARER_TOKEN = "***REMOVED***"
