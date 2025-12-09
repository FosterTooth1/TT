# Usamos una imagen oficial de Python como base
FROM python:3.9-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos dependencias del sistema operativo que puedan ser necesarias
# para tu librería de C++ y otras bibliotecas de Python.
RUN apt-get update && apt-get install -y build-essential

# Copiamos el archivo de requerimientos primero para aprovechar el cache de Docker
COPY requirements.txt requirements.txt

# Instalamos las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todos los archivos de tu proyecto al directorio de trabajo del contenedor
COPY . .

# Exponemos el puerto 8080, que es el puerto estándar que Cloud Run espera
EXPOSE 8080

# El comando para ejecutar tu aplicación usando Gunicorn
# Escuchará en todas las interfaces de red (0.0.0.0) en el puerto 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "Main:app"]