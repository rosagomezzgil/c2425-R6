# config.py
"""
Módulo: configuración de credenciales para autenticación con la API de twitter.

Este script contiene las claves necesarias para autenticar el acceso a la API de twitter
(v2) usando la librería `tweepy`. Es utilizado por los módulos de extracción de datos
que requieren conexión a dicha API.

IMPORTANTE:
Las claves se leen de variables de entorno (por ejemplo, desde un archivo `.env`
que no se sube al repositorio). Ver `.env.example`.
"""

import os

API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
