"""
test.py

Este script tiene como objetivo validar el correcto funcionamiento de la API REST implementada en app.py.
A través de solicitudes HTTP realizadas con la librería requests, se comprueba tanto la accesibilidad del
servidor como la capacidad de generar predicciones y manejar entradas erróneas de forma adecuada.

# ====================================================================
# CONFIGURACIÓN DEL SISTEMA DE LOGGING
# ====================================================================
# Configura el sistema de registro de eventos para monitoreo y debugging.
#
# Nivel: INFO - Registra eventos informativos, advertencias y errores
# Formato: Timestamp + Nivel + Mensaje
# Destinos:
#   - app.log: Persistencia en disco para auditoría
#   - StreamHandler: Consola para monitoreo en tiempo real
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_app.log"),
        logging.StreamHandler(),
    ],
)

logging.info("Iniciando pruebas de la API REST")

# ====================================================================
# FLUJO GENERAL DEL SCRIPT
# ====================================================================
# Flujo general del script:
1. Verificación del endpoint raíz (GET /):
   - Se envía una solicitud GET al servidor para confirmar que la API está activa.
   - Se espera como respuesta un mensaje de estado en formato JSON.

2. Prueba de predicción válida (POST /predict):
   - Se construye un JSON con la clave "features" que contiene una lista de 30 valores numéricos,
     correspondientes a las características del dataset Breast Cancer.
   - Se envía una solicitud POST al endpoint de predicción.
   - Se espera como respuesta un JSON con la clave "prediction" cuyo valor es 0 (maligno) o 1 (benigno).

3. Prueba de manejo de errores (POST /predict):
   - Se envía un JSON mal formado, sin la clave "features".
   - Se espera que la API responda con un mensaje de error descriptivo y un código HTTP 400,
     lo que demuestra la robustez del servicio ante entradas inválidas.

Este script debe ejecutarse en una terminal separada mientras la aplicación Flask (app.py) está en ejecución.
Su propósito es verificar de manera automatizada que los componentes del sistema interactúan correctamente,
asegurando así la validez del proceso de despliegue del modelo.
"""

# Paquetes necesarios
# pip install requests

import requests
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("test_app.log"),
        logging.StreamHandler(),
    ],
)

logging.info("Iniciando pruebas de la API REST")
# ====================================================================
# 1. Probar endpoint raíz
# ====================================================================
res = requests.get("http://127.0.0.1:5000/")
logging.info("GET /:", res.json())

# ====================================================================
# 2. Probar endpoint predict con datos válidos
# ====================================================================
ejemplo = {
    "features": [
        14.2,
        20.3,
        92.4,
        600.5,
        0.1,
        0.2,
        0.3,
        0.1,
        0.2,
        0.05,
        0.3,
        1.0,
        2.0,
        30.0,
        0.01,
        0.1,
        0.05,
        0.01,
        0.05,
        0.01,
        15.0,
        25.0,
        100.0,
        700.0,
        0.12,
        0.4,
        0.6,
        0.2,
        0.3,
        0.08,
    ]
}
res = requests.post("http://127.0.0.1:5000/predict", json=ejemplo)
logging.info(f"POST /predict: {res.json()}")
# ====================================================================
# 3. Probar con error en el input
# ====================================================================
# a) Clave incorrecta
# ====================================================================
res = requests.post("http://127.0.0.1:5000/predict", json={"dato": [1, 2, 3]})
logging.info(f"POST /predict con error: {res.json()}")

# ====================================================================
# b) Tipo de dato incorrecto
# ====================================================================
res = requests.post(
    "http://127.0.0.1:5000/predict", json={"features": "dato_incorrecto"}
)
logging.info(f"POST /predict con error:, {res.json()}")

# ====================================================================
# c) Datos corruptos (mezcla de tipos)
# ====================================================================
res = requests.post("http://127.0.0.1:5000/predict", json={})
logging.info(f"POST /predict con error:, {res.json()}")

corrupto = {
    "features": [
        14.2,
        20.3,
        92.4,
        600.5,
        0.1,
        0.2,
        0.3,
        0.1,
        0.2,
        0.05,
        0.3,
        1.0,
        2.0,
        30.0,
        0.01,
        0.1,
        0.05,
        0.01,
        0.05,
        0.01,
        15.0,
        25.0,
        100.0,
        "a",
        0.12,
        0.4,
        0.6,
        0.2,
        0.3,
        0.08,
    ]
}

res = requests.post("http://127.0.0.1:5000/predict", json=corrupto)
logging.info(f"POST /predict con error: {res.json()}")

# ====================================================================
# 4. Pruebas completadas con éxito
# ====================================================================
logging.info("Pruebas completadas con éxito")
logging.info("Fin de las pruebas")
logging.info("Henzo Alejandro Arrué Muñoz")
logging.info("Version: 1.0.0")
logging.info("Fecha: 2025")
