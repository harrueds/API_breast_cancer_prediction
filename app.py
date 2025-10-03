"""API REST para Predicción de Cáncer de Mama - Breast Cancer Wisconsin.

Este módulo implementa una API RESTful basada en Flask para servir predicciones
de clasificación binaria de tumores mamarios utilizando un modelo de Machine Learning
pre-entrenado (Regresión Logística).

Arquitectura:
    La aplicación sigue el patrón de microservicios para inferencia de modelos,
    separando el entrenamiento (training.py) del despliegue (app.py). El modelo
    serializado se carga en memoria al iniciar la aplicación para optimizar
    el tiempo de respuesta.

Endpoints:
    GET  /        : Health check - Verifica el estado de la API
    POST /predict : Realiza predicciones sobre datos de entrada

Formatos de Entrada Soportados:
    1. Lista de características (array numérico):
       {"features": [valor1, valor2, ..., valor30]}
    
    2. Diccionario con nombres de características:
       {"feature_1": valor1, "feature_2": valor2, ..., "feature_30": valor30}

Formato de Salida:
    {
        "predicción": "Benigno" | "Maligno",
        "probabilidad": float  # Confianza del modelo [0.0, 1.0]
    }

Clasificación:
    - Clase 0 (Benigno): Tumor no canceroso
    - Clase 1 (Maligno): Tumor canceroso

Requisitos del Sistema:
    - Python >= 3.7
    - Flask >= 2.0.0
    - scikit-learn >= 1.0.0
    - pandas >= 1.3.0
    - joblib >= 1.0.0

Ejecución:
    $ python app.py
    
    La API estará disponible en: http://0.0.0.0:5000

Ejemplo de Uso:
    $ curl -X POST http://localhost:5000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"features": [17.99, 10.38, 122.8, ...]}'

Seguridad:
    ADVERTENCIA: Esta implementación está configurada con debug=True y es
    adecuada únicamente para entornos de desarrollo. Para producción, se
    recomienda:
    - Desactivar modo debug
    - Implementar autenticación y autorización
    - Utilizar HTTPS
    - Configurar rate limiting
    - Validar y sanitizar todas las entradas

Logging:
    Los eventos se registran en:
    - app.log: Archivo persistente con historial completo
    - stdout: Salida estándar para monitoreo en tiempo real

Autor: Henzo Alejandro Arrué Muñoz
Versión: 1.0.0
Fecha: 2025
"""

from flask import Flask, request, Response
import joblib
import logging
import pandas as pd
import json

# ============================================================================
# CONFIGURACIÓN DEL SISTEMA DE LOGGING
# ============================================================================
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
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)

# ============================================================================
# INICIALIZACIÓN DE LA APLICACIÓN FLASK
# ============================================================================
# Crea la instancia de la aplicación Flask y carga el modelo pre-entrenado.
#
# El modelo se carga una sola vez al iniciar la aplicación (patrón singleton)
# para evitar la sobrecarga de deserialización en cada predicción.
app = Flask(__name__)

# ============================================================================
# CARGA DEL MODELO DE MACHINE LEARNING
# ============================================================================
# Deserializa el modelo de Regresión Logística entrenado previamente.
#
# Archivo: modelo.pkl
# Formato: Joblib pickle (optimizado para objetos scikit-learn)
# Contenido: Modelo LogisticRegression entrenado con dataset Breast Cancer Wisconsin
modelo = joblib.load("modelo.pkl")
logging.info("Modelo de clasificación cargado exitosamente en memoria")


def respuesta_json(data, status=200):
    """Genera una respuesta HTTP en formato JSON con codificación UTF-8.

    Esta función auxiliar construye respuestas HTTP estandarizadas en formato JSON,
    preservando caracteres Unicode (tildes, ñ, etc.) sin escaparlos a ASCII.
    El formato incluye indentación para legibilidad humana y un salto de línea final.

    Args:
        data (dict): Diccionario Python a serializar como JSON.
                    Debe ser serializable por json.dumps().
        status (int, optional): Código de estado HTTP. Por defecto 200 (OK).
                               Códigos comunes:
                               - 200: Éxito
                               - 400: Error del cliente (Bad Request)
                               - 500: Error del servidor (Internal Server Error)

    Returns:
        flask.Response: Objeto Response de Flask con:
                       - Content-Type: application/json; charset=utf-8
                       - Body: JSON serializado con indentación de 4 espacios
                       - Status: Código HTTP especificado

    Ejemplo:
        >>> respuesta_json({"mensaje": "Operación exitosa"}, 200)
        <Response [200]>

        >>> respuesta_json({"error": "Parámetro inválido"}, 400)
        <Response [400]>

    Notas:
        - ensure_ascii=False preserva caracteres UTF-8 nativos
        - indent=4 mejora la legibilidad para debugging
        - El salto de línea final facilita la lectura en terminal
        - Reduce el tamaño del payload al evitar secuencias de escape
    """
    return Response(
        response=json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        status=status,
        mimetype="application/json; charset=utf-8",
    )


@app.route("/", methods=["GET"])
def estado():
    """Endpoint de health check para verificar el estado de la API.

    Este endpoint implementa un health check básico para monitoreo de disponibilidad.
    Se utiliza típicamente en:
    - Sistemas de orquestación (Kubernetes liveness/readiness probes)
    - Balanceadores de carga
    - Herramientas de monitoreo (Prometheus, Nagios, etc.)

    Ruta:
        GET /

    Parámetros:
        Ninguno

    Returns:
        flask.Response: JSON con código 200 (OK) indicando que la API está operativa.

                       Estructura de respuesta:
                       {
                           "status": "OK",
                           "mensaje": "API en línea y a la espera"
                       }

    Códigos de Estado:
        200: API operativa y lista para recibir peticiones

    Ejemplo de Uso:
        $ curl http://localhost:5000/
        {
            "status": "OK",
            "mensaje": "API en línea y a la espera"
        }

    Notas:
        - Este endpoint no verifica la integridad del modelo cargado
        - Para un health check completo, considere verificar modelo.predict()
        - El acceso se registra en el sistema de logging
    """
    logging.info("Health check solicitado - Endpoint raíz accedido")
    return respuesta_json({"status": "OK", "mensaje": "API en línea y a la espera"})


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint de inferencia para clasificación de tumores mamarios.
    
    Este endpoint recibe características de un tumor y retorna una predicción binaria
    (Benigno/Maligno) junto con la probabilidad asociada, utilizando el modelo de
    Regresión Logística pre-entrenado.
    
    Ruta:
        POST /predict
    
    Content-Type:
        application/json
    
    Formatos de Entrada Aceptados:
        
        Formato 1 - Array de características (sin nombres):
        {
            "features": [
                17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
                0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
                0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
                184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
            ]
        }
        
        Formato 2 - Diccionario con nombres de características:
        {
            "mean_radius": 17.99,
            "mean_texture": 10.38,
            ...
            "worst_fractal_dimension": 0.1189
        }
    
    Validaciones:
        - El payload debe ser JSON válido
        - Se requieren exactamente 30 características numéricas
        - Los valores deben ser numéricos (int o float)
    
    Returns:
        flask.Response: JSON con la predicción y probabilidad asociada.
        
        Respuesta exitosa (200):
        {
            "predicción": "Benigno" | "Maligno",
            "probabilidad": 0.9542  # float en rango [0.0, 1.0]
        }
        
        Respuesta de error (400):
        {
            "error": "Descripción del error de validación"
        }
        
        Respuesta de error (500):
        {
            "error": "Error interno en el servidor"
        }
    
    Códigos de Estado HTTP:
        200: Predicción exitosa
        400: Error de validación en los datos de entrada
        500: Error interno del servidor durante la inferencia
    
    Proceso de Inferencia:
        1. Validación del formato JSON
        2. Validación de la estructura de datos (30 características)
        3. Conversión a DataFrame de pandas
        4. Predicción con modelo.predict()
        5. Cálculo de probabilidad con modelo.predict_proba()
        6. Mapeo de clase numérica a etiqueta textual
        7. Serialización de respuesta JSON
    
    Ejemplo de Uso:
        $ curl -X POST http://localhost:5000/predict \\
          -H "Content-Type: application/json" \\
          -d '{"features": [17.99, 10.38, ..., 0.1189]}'
        
        Respuesta:
        {
            "predicción": "Maligno",
            "probabilidad": 0.9542
        }
    
    Interpretación de Resultados:
        - Probabilidad > 0.5: Alta confianza en la predicción
        - Probabilidad ≈ 0.5: Baja confianza, caso ambiguo
        - "Benigno": Tumor no canceroso (clase 0)
        - "Maligno": Tumor canceroso (clase 1)
    
    Consideraciones Clínicas:
        ADVERTENCIA: Este sistema es una herramienta de apoyo diagnóstico y NO
        debe utilizarse como único criterio para decisiones médicas. Siempre debe
        ser complementado con:
        - Evaluación clínica profesional
        - Estudios histopatológicos
        - Análisis de imágenes médicas
        - Contexto clínico del paciente
    
    Manejo de Errores:
        - Todos los errores se registran en el sistema de logging
        - Los errores de validación retornan mensajes descriptivos
        - Los errores internos ocultan detalles técnicos al cliente
    
    Raises:
        No propaga excepciones. Todos los errores se capturan y retornan
        como respuestas HTTP con códigos de estado apropiados.
    """
    try:
        # ====================================================================
        # 1. VALIDACIÓN DE PAYLOAD JSON
        # ====================================================================
        # Extrae y valida que el cuerpo de la petición contenga JSON válido.
        data = request.get_json()
        if not data:
            logging.warning("Solicitud rechazada: Payload JSON vacío o nulo")
            return respuesta_json({"error": "Se esperaba un JSON con datos"}, 400)

        df = None

        # ====================================================================
        # 2. PROCESAMIENTO DE FORMATO 1: ARRAY DE CARACTERÍSTICAS
        # ====================================================================
        # Valida y procesa entrada en formato {"features": [v1, v2, ..., v30]}
        if "features" in data:
            features = data.get("features")

            # Validación de tipo y dimensionalidad
            if not isinstance(features, list) or len(features) != 30:
                logging.warning(
                    f"Validación fallida: Se esperaban 30 features en lista, "
                    f"recibidas {len(features) if features else 0}"
                )
                return respuesta_json(
                    {
                        "error": "Formato inválido, se requieren 30 características en la lista 'features'"
                    },
                    400,
                )

            # Conversión a DataFrame para compatibilidad con scikit-learn
            df = pd.DataFrame([features])

        # ====================================================================
        # 3. PROCESAMIENTO DE FORMATO 2: DICCIONARIO CON NOMBRES
        # ====================================================================
        # Valida y procesa entrada en formato {"feature_1": v1, "feature_2": v2, ...}
        else:
            df = pd.DataFrame([data])

            # Validación de dimensionalidad
            if df.shape[1] != 30:
                logging.warning(
                    f"Validación fallida: Se esperaban 30 features con nombre, "
                    f"recibidas {df.shape[1]}"
                )
                return respuesta_json(
                    {
                        "error": "Formato inválido, se requieren 30 características con nombre"
                    },
                    400,
                )

        # ====================================================================
        # 4. INFERENCIA DEL MODELO
        # ====================================================================
        # Realiza la predicción y calcula la probabilidad asociada.

        # Predicción de clase (0 = Benigno, 1 = Maligno)
        pred = modelo.predict(df)[0]

        # Probabilidad de la clase predicha
        # predict_proba retorna [P(clase_0), P(clase_1)]
        prob = modelo.predict_proba(df)[0][pred]

        # Mapeo de clase numérica a etiqueta semántica
        resultado = "Maligno" if pred == 1 else "Benigno"

        # ====================================================================
        # 5. LOGGING Y RESPUESTA
        # ====================================================================
        logging.info(
            f"Predicción exitosa: {resultado} (clase {pred}) con probabilidad={prob:.4f}"
        )
        return respuesta_json({"predicción": resultado, "probabilidad": float(prob)})

    except Exception as e:
        # ====================================================================
        # 6. MANEJO DE ERRORES INESPERADOS
        # ====================================================================
        # Captura cualquier excepción no anticipada durante el proceso de inferencia.
        logging.error(f"Error crítico durante la predicción: {type(e).__name__}: {e}")
        return respuesta_json({"error": "Error interno en el servidor"}, 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
