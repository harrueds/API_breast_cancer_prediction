"""Script de Entrenamiento de Modelo de Clasificación - Breast Cancer Wisconsin.

Este módulo implementa el pipeline completo de entrenamiento y serialización de un modelo
de clasificación binaria para la detección de cáncer de mama utilizando el dataset
Breast Cancer Wisconsin Diagnostic.

Proceso:
    1. Carga del dataset desde scikit-learn
    2. División de datos en conjuntos de entrenamiento (80%) y prueba (20%)
    3. Entrenamiento de un modelo de Regresión Logística
    4. Serialización del modelo entrenado para su posterior despliegue
    5. Registro de eventos mediante logging

Dataset:
    Nombre: Breast Cancer Wisconsin Diagnostic
    Características: 30 variables numéricas derivadas de imágenes digitalizadas
                    (media, error estándar y peor valor de 10 medidas morfológicas)
    Clases: Binaria (0 = maligno, 1 = benigno)
    Muestras: 569 instancias

Modelo:
    Algoritmo: Regresión Logística
    Hiperparámetros: max_iter=5000 (para garantizar convergencia del optimizador)
    Propósito: Clasificación binaria de tumores mamarios

Salida:
    - modelo.pkl: Modelo serializado con joblib
    - training.log: Registro de eventos del proceso de entrenamiento

Requisitos:
    scikit-learn>=1.0.0
    joblib>=1.0.0

Ejemplo de uso:
    $ python training.py

Autor: Machine Learning Kibernum
Fecha: 2025
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import logging

# =========================
# Configuración de Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)

# ============================================================================
# 1. CARGA DEL DATASET
# ============================================================================
# Carga el dataset Breast Cancer Wisconsin Diagnostic desde scikit-learn.
#
# Variables:
#   X: Matriz de características (569 muestras × 30 características)
#   y: Vector objetivo con etiquetas de clase (0 = maligno, 1 = benigno)
#
# El dataset se carga directamente en formato NumPy arrays para su procesamiento.
X, y = load_breast_cancer(return_X_y=True)
logging.info(
    "Dataset Breast Cancer Wisconsin cargado: %d muestras, %d características",
    X.shape[0],
    X.shape[1],
)

# ============================================================================
# 2. DIVISIÓN DE DATOS EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
# ============================================================================
# Separa los datos en conjuntos de entrenamiento (80%) y prueba (20%).
#
# Parámetros:
#   test_size=0.2: Reserva el 20% de los datos para evaluación del modelo
#   random_state=42: Semilla aleatoria para garantizar reproducibilidad
#
# La división es aleatoria pero reproducible, permitiendo comparaciones consistentes
# entre diferentes ejecuciones del script.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logging.info(
    "Datos divididos - Entrenamiento: %d muestras, Prueba: %d muestras",
    X_train.shape[0],
    X_test.shape[0],
)

# ============================================================================
# 3. ENTRENAMIENTO DEL MODELO
# ============================================================================
# Inicializa y entrena un modelo de Regresión Logística.
#
# Hiperparámetros:
#   max_iter=5000: Número máximo de iteraciones para el optimizador (LBFGS por defecto)
#                  Valor incrementado desde el default (100) para asegurar convergencia
#
# El método fit() ajusta los coeficientes del modelo mediante optimización iterativa,
# minimizando la función de pérdida logística (log-loss) sobre los datos de entrenamiento.
model = LogisticRegression(max_iter=5000)
logging.info("Iniciando entrenamiento del modelo de Regresión Logística...")

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
f1 = f1_score(y_test, model.predict(X_test))
precision = precision_score(y_test, model.predict(X_test))
recall = recall_score(y_test, model.predict(X_test))


logging.info("=" * 60)
logging.info("Modelo entrenado exitosamente")
logging.info("=" * 60)
logging.info(f"Accuracy={accuracy:.4f}")
logging.info(f"F1-Score={f1:.4f}")
logging.info(f"Precision={precision:.4f}")
logging.info(f"Recall={recall:.4f}")
logging.info("=" * 60)


# ============================================================================
# 4. SERIALIZACIÓN DEL MODELO ENTRENADO
# ============================================================================
# Guarda el modelo entrenado en disco utilizando joblib para su posterior uso.
#
# Detalles:
#   Formato: Pickle optimizado para objetos NumPy/scikit-learn
#   Archivo: modelo.pkl
#   Propósito: Permite reutilizar el modelo en producción sin reentrenamiento
#
# El modelo serializado puede ser cargado posteriormente con joblib.load('modelo.pkl')
# para realizar predicciones en nuevos datos.
joblib.dump(model, "modelo.pkl")
logging.info("Modelo serializado y guardado como 'modelo.pkl'")

# ============================================================================
# 5. CONFIRMACIÓN DE FINALIZACIÓN
# ============================================================================
# Registra la finalización exitosa del pipeline de entrenamiento.
logging.info("=" * 60)
logging.info("Pipeline de entrenamiento completado exitosamente")
logging.info("Modelo disponible en: modelo.pkl")
logging.info("Logs disponibles en: training.log")
logging.info("=" * 60)
