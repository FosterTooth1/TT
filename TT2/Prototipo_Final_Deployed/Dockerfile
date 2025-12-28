# Usamos una imagen oficial de Python como base
FROM python:3.9-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos dependencias del sistema operativo que puedan ser necesarias
# para la librería de C++ y otras bibliotecas de Python.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiamos el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalamos las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todos los archivos de tu proyecto al directorio de trabajo del contenedor
COPY . .

# Exponemos el puerto 8080
EXPOSE 8080

# Establecemos variables de entorno
ENV FLASK_APP=app.main
ENV PYTHONUNBUFFERED=1

# El comando para ejecutar tu aplicación usando Gunicorn
# Escuchará en todas las interfaces de red (0.0.0.0) en el puerto 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "app.main:app"]