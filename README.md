# API REST para Predicción de Cáncer de Mama

Este proyecto implementa una **API REST en Python con Flask** para predecir la probabilidad de cáncer de mama a partir de parámetros numéricos.  
La solución está diseñada con un **enfoque académico y profesional**, integrando metodologías de *Machine Learning*, contenedorización con **Docker**, y despliegue automatizado mediante **GitHub Actions (CI/CD)**.

---

## Objetivos

- Desarrollar una API REST confiable y ligera para predicciones de cáncer de mama.
- Contenerizar la aplicación para portabilidad y despliegue en múltiples entornos.
- Automatizar pruebas e integración continua utilizando **CI/CD en GitHub Actions**.
- Asegurar que la API acepte **únicamente entradas numéricas válidas**, coherentes con el modelo entrenado.

---

## Estructura del proyecto

```bash
/modular/   # Directorio principal del proyecto
├── training.py # Script de entrenamiento y guardado del modelo
├── app.py # API REST con Flask para exponer el modelo
├── test_app.py # Script de pruebas unitarias y de integración
├── modelo.pkl # Modelo entrenado (generado por training.py)
├── requirements.txt # Dependencias necesarias
├── Dockerfile # Definición de la imagen Docker
├── .github/
├   ├── workflows/
├   └── ci-cd.yml # Configuración de CI/CD con GitHub Actions
```

---

## Requisitos

- **Python 3.10+**
- Librerías especificadas en `requirements.txt`
- **Docker Engine 20+**
- Cuenta en **Docker Hub** (para subir la imagen)
- Repositorio en **GitHub** con CI/CD habilitado

Instalación de dependencias locales:

```bash
pip install -r requirements.txt
```

---

### Uso con Docker

#### Construcción de la imagen

Para construir la imagen Docker se debe ejecutar el siguiente comando:

```bash
docker build -t henzo_arrue_mod10 .
```

#### Ejecución de un contenedor

```bash
docker run -d -p 5000:5000 henzo_arrue_mod10
```

#### Acceso a la API

```bash
curl http://127.0.0.1:5000
```

### Pruebas

Ejecución manual de pruebas con Python:

```bash
python test_app.py
```

#### Pruebas automáticas incluidas en el flujo de CI/CD

### Flujo CI/CD (GitHub Actions)

El archivo .github/workflows/ci-cd.yml automatiza:

- Construcción de la imagen Docker.
- Ejecución de pruebas automáticas en cada push o pull request a la rama main.
- Subida de la imagen a Docker Hub al fusionar en main.

---

## Nota académica

El proyecto representa un caso de estudio en la integración de Machine Learning aplicado a la salud con herramientas modernas de despliegue en la nube. Combina:

- Buenas prácticas de ingeniería de software (modularidad, pruebas automatizadas).
- Contenerización (Docker) para reproducibilidad.
- Automatización CI/CD para asegurar calidad continua en entornos colaborativos.

---

## Contacto

Si tienes alguna pregunta o sugerencia, no dudes en contactarme:

- **Email:** [henzoarrue@gmail.com](mailto:henzoarrue@gmail.com)
- **LinkedIn:** [https://www.linkedin.com/in/henzo-arrue/](https://www.linkedin.com/in/henzo-arru%C3%A9-mu%C3%B1oz/)
- **GitHub:** [https://github.com/harrueds](https://github.com/harrueds)
