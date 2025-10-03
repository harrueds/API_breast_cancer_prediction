# ==============================================================================
# Dockerfile para API de Predicción de Cáncer de Mama
# ==============================================================================
# Este Dockerfile construye una imagen de contenedor que incluye:
#   - API REST implementada con Flask
#   - Modelo de Machine Learning pre-entrenado
#   - Todas las dependencias necesarias para ejecutar predicciones
#
# Características:
#   - Imagen base: Python 3.10 slim (optimizada para tamaño reducido)
#   - Puerto expuesto: 5000
#   - Entrenamiento del modelo durante el build (modelo.pkl incluido en la imagen)
#
# Uso:
#   docker build -t henzo_arrue_mod10 .
#   docker run -p 5000:5000 --name henzo_arrue_mod10 henzo_arrue_mod10

# ------------------------------------------------------------------------------
# ETAPA 1: Selección de la imagen base
# ------------------------------------------------------------------------------
# Se utiliza python:3.10-slim por las siguientes razones:
#   - Versión estable y compatible con las librerías de ML
#   - Variante "slim" reduce el tamaño de la imagen (~120MB vs ~900MB de la versión completa)
#   - Incluye las herramientas esenciales de Python sin paquetes innecesarios
FROM python:3.10-slim

# ------------------------------------------------------------------------------
# ETAPA 2: Configuración del directorio de trabajo
# ------------------------------------------------------------------------------
# Establece /app como directorio de trabajo dentro del contenedor
# Todos los comandos subsecuentes se ejecutarán en este directorio
WORKDIR /app
# ============================================================================
# VARIABLES DE ENTORNO
# ============================================================================
# - PYTHONDONTWRITEBYTECODE evita archivos .pyc innecesarios
# - PYTHONUNBUFFERED fuerza logs en tiempo real (stdout/stderr)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# ETAPA 3: Instalación de dependencias de Python
# ------------------------------------------------------------------------------
# Se copia requirements.txt ANTES que el resto del código para aprovechar
# el sistema de caché de Docker. Si requirements.txt no cambia, esta capa
# se reutiliza en builds posteriores, acelerando el proceso de construcción
COPY requirements.txt .

# Instalación de dependencias del sistema operativo necesarias para compilar
# paquetes de Python que contienen extensiones en C/C++ (numpy, scikit-learn)
# Componentes instalados:
#   - build-essential: herramientas de compilación básicas
#   - gcc: compilador de C
#   - g++: compilador de C++
# Optimizaciones aplicadas:
#   - --no-install-recommends: instala solo paquetes esenciales
#   - rm -rf /var/lib/apt/lists/*: elimina caché de apt para reducir tamaño
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Instalación de dependencias de Python desde requirements.txt
# Flags utilizados:
#   - --no-cache-dir: no guarda archivos de caché de pip (reduce tamaño de imagen)
RUN pip install --upgrade pip && pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# ------------------------------------------------------------------------------
# ETAPA 4: Copia del código fuente
# ------------------------------------------------------------------------------
# Copia todos los archivos del proyecto al contenedor
# Incluye: app.py, training.py, test_api.py y cualquier otro archivo necesario
# Nota: Los archivos listados en .dockerignore no serán copiados
COPY . .

# ------------------------------------------------------------------------------
# ETAPA 5: Entrenamiento del modelo
# ------------------------------------------------------------------------------
# Ejecuta el script de entrenamiento durante la construcción de la imagen
# Ventajas de entrenar en build-time vs run-time:
#   - El modelo.pkl queda embebido en la imagen
#   - Contenedores arrancan instantáneamente sin necesidad de entrenar
#   - Garantiza que todos los contenedores usan el mismo modelo
# Resultado: genera el archivo modelo.pkl en /app/
RUN python training.py

# ------------------------------------------------------------------------------
# ETAPA 6: Configuración de red
# ------------------------------------------------------------------------------
# Expone el puerto 5000 para que el servidor Flask sea accesible
# Nota: EXPOSE es documentativo; el puerto debe mapearse al ejecutar el contenedor
# Ejemplo: docker run -p 8080:5000 <imagen> mapea el puerto 5000 del contenedor al 8080 del host
EXPOSE 5000

# ------------------------------------------------------------------------------
# ETAPA 7: Comando de ejecución
# ------------------------------------------------------------------------------
# Define el comando que se ejecutará cuando el contenedor inicie
# Inicia el servidor Flask que expone la API de predicción
# Formato exec (lista): preferido porque permite señales del sistema (SIGTERM, SIGINT)
CMD ["python", "app.py"]