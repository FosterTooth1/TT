# LogistiClima - Sistema de Optimización de Rutas Logísticas

## Descripción
Sistema web que integra predicciones climáticas para optimizar rutas de entrega a naves industriales usando el algoritmo de Recocido Simulado.

## Características
- **Optimización de rutas** con algoritmo de Recocido Simulado
- **Predicciones climáticas** en tiempo real usando Machine Learning
- **Visualización geográfica** de rutas optimizadas
- **Interfaz web** para selección de destinos y visualización de resultados
- **Panel de administración** para configurar parámetros del algoritmo
- **Sistema de autenticación** con registro e inicio de sesión
- **Historial de rutas** guardadas por usuario

## Instalación

### Requisitos
- Python 3.9+
- Docker (para despliegue en producción)
- Base de datos MySQL (Cloud SQL en producción)

### Variables de Entorno Requeridas
```
SECRET_KEY=clave_secreta_para_sesiones
DB_USER=usuario_base_datos
DB_PASS=contraseña_base_datos
DB_NAME=nombre_base_datos
DB_HOST=/cloudsql/proyecto:region:instancia
WEATHER_API_KEY=api_key_weatherapi
```

### Pasos de instalación local

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Ejecutar el servidor:**
```bash
python app/main.py
```

3. **Abrir en el navegador:**
   - URL: `http://localhost:5000`

### Despliegue con Docker

1. **Construir imagen:**
```bash
docker build -t logisticlima .
```

2. **Ejecutar contenedor:**
```bash
docker run -p 8080:8080 logisticlima
```

## Uso

### Flujo Principal
1. **Seleccionar Naves**: Elige al menos 5 naves industriales de la lista
2. **Seleccionar Inicio**: Indica desde qué nave comenzar la ruta
3. **Generar Ruta**: El sistema calcula la ruta optimizada considerando condiciones climáticas
4. **Visualizar Resultado**: Ve la ruta en el mapa interactivo con comparativa de rutas con penalizaciones y sin penalizaciones
5. **Guardar Ruta**: Los usuarios autenticados pueden guardar sus rutas

### Funcionalidades
- **Selección de destinos**: Checkbox para elegir naves industriales
- **Optimización de rutas**: Considera distancias reales y penalizaciones climáticas
- **Comparativa de rutas**: Muestra ruta con y sin penalización climática
- **Visualización en mapa**: Marcadores y ruta optimizada usando Leaflet
- **Información detallada**: Tabla con orden de visita y condiciones climáticas
- **Gráfica de convergencia**: Muestra la evolución del fitness durante la optimización

## Estructura del Proyecto

```
├── app/
│   ├── main.py                     # Servidor Flask y lógica principal
│   └── prediccion_clima.pkl        # Modelo de ML entrenado
├── config/
│   └── parametros.json             # Parámetros del algoritmo
├── data/
│   ├── Naves_Industriales.csv      # Base de datos de naves
│   └── Matriz_Distancias_Carretera.csv  # Matriz de distancias
├── lib/
│   ├── recocido.dll                # Biblioteca C++ (Windows)
│   └── librecocido.so              # Biblioteca C++ (Linux)
├── static/
│   ├── animations/                 # Animaciones Lottie
│   ├── css/                        # Estilos CSS
│   │   ├── estilo_auth.css
│   │   ├── estilo_IniciarSesion.css
│   │   ├── estilo_MejorRuta.css
│   │   ├── estilo_NuevaCuenta.css
│   │   ├── estilo_RutasRecientes.css
│   │   └── estilo_SeleccionarNaves.css
│   ├── images/                     # Imágenes
│   │   └── logo.jpg
│   │   ├── marker-icon-2x-green.png
│   │   └── marker-shadow.png
│   └── js/                         # JavaScript
│       ├── js_IniciarSesion.js
│       ├── js_MejorRuta.js
│       ├── js_NuevaCuenta.js
│       ├── js_RutasRecientes.js
│       └── js_SeleccionarNaves.js
├── templates/                      # Plantillas HTML
│   ├── IniciarSesion.html          # Inicio de sesión
│   ├── MejorRuta.html              # Visualización de resultados
│   ├── NuevaCuenta.html            # Registro de usuarios
│   ├── panel_admin.html            # Panel de administración
│   ├── RutasRecientes.html         # Historial de rutas
│   └── SeleccionarNaves.html       # Selección de destinos
├── Dockerfile                      # Configuración Docker
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

## APIs Disponibles

### Públicas
- `GET /api/naves` - Obtener lista de naves industriales
- `POST /api/generar-ruta` - Generar ruta optimizada
  - Body: `{"indices": [0, 1, 2, ...], "indice_inicio": 0}`
  - Response: Rutas optimizadas con y sin penalización climática

### Autenticación
- `POST /registrar_usuario` - Registrar nuevo usuario
- `POST /login_usuario` - Iniciar sesión
- `POST /cerrar-sesion` - Cerrar sesión

### Rutas (requiere autenticación)
- `GET /obtener_rutas` - Obtener rutas guardadas del usuario
- `POST /guardar-ruta` - Guardar una ruta
- `DELETE /eliminar-ruta/<id>` - Eliminar una ruta
- `GET /regenerar-ruta/<id>` - Regenerar una ruta guardada

### Administración (requiere rol admin)
- `GET /panel_admin` - Panel de configuración
- `POST /actualizar_parametros` - Actualizar parámetros del algoritmo

## Tecnologías Utilizadas

- **Backend**: Python 3.9, Flask, Gunicorn, Pandas, NumPy
- **Base de datos**: MySQL (Cloud SQL)
- **Algoritmo**: Recocido Simulado (implementado en C++)
- **Machine Learning**: Random Forest para predicciones climáticas
- **APIs Externas**: WeatherAPI para datos climáticos en tiempo real
- **Frontend**: HTML5, CSS3, JavaScript
- **Contenedores**: Docker
- **Despliegue**: Google Cloud Run

## Notas de Desarrollo

- El sistema requiere conexión a internet para la API de clima
- Las predicciones climáticas se realizan en paralelo para mejor rendimiento
- El algoritmo de optimización usa la biblioteca C++ para mayor velocidad
- La base de datos de naves industriales está enfocada en el Estado de México
- Los parámetros del algoritmo son configurables desde el panel de administración