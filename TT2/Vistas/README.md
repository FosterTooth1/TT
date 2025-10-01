# LogistiClima - Sistema de Optimización de Rutas Logísticas

## Descripción
Sistema web que integra predicciones climáticas para optimizar rutas de entrega a naves industriales usando el algoritmo de Recocido Simulado.

## Características
- **Optimización de rutas** con algoritmo de Recocido Simulado
- **Predicciones climáticas** en tiempo real usando Machine Learning
- **Visualización geográfica** de rutas optimizadas
- **Interfaz web** para selección de destinos y visualización de resultados

## Instalación

### Requisitos
- Python 3.8+
- Archivos de datos: `Naves_Industriales.csv`, `Matriz_Distancias_Carretera.csv`, `prediccion_clima.pkl`
- Bibliotecas C++: `recocido.dll` (Windows) o `librecocido.so` (Linux)

### Pasos de instalación

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Verificar archivos necesarios:**
   - `Naves_Industriales.csv` - Base de datos de naves industriales
   - `Matriz_Distancias_Carretera.csv` - Matriz de distancias entre naves
   - `prediccion_clima.pkl` - Modelo de Machine Learning entrenado
   - `recocido.dll` / `librecocido.so` - Biblioteca del algoritmo de optimización

3. **Ejecutar el servidor:**
```bash
python Main.py web
```

4. **Abrir en el navegador:**
   - URL: `http://localhost:5000`

## Uso

### Flujo Principal
1. **Seleccionar Naves**: Elige al menos 5 naves industriales de la lista
2. **Generar Ruta**: El sistema calcula la ruta optimizada considerando condiciones climáticas
3. **Visualizar Resultado**: Ve la ruta en el mapa interactivo y la tabla de detalles

### Funcionalidades
- **Selección de destinos**: Checkbox para elegir naves industriales
- **Optimización automática**: Considera distancias reales y condiciones climáticas
- **Visualización en mapa**: Marcadores y ruta optimizada usando Leaflet
- **Información detallada**: Tabla con orden de visita y condiciones climáticas

## Estructura del Proyecto

```
├── Main.py                          # Servidor Flask y lógica principal
├── templates/                       # Plantillas HTML
│   ├── SeleccionarNaves.html       # Vista principal de selección
│   ├── MejorRuta.html              # Vista de resultados
│   ├── IniciarSesion.html          # Autenticación
│   ├── NuevaCuenta.html            # Registro
│   └── RutasRecientes.html         # Historial (futuro)
├── static/                         # Archivos estáticos
│   ├── estilo/                     # Archivos CSS
│   ├── js/                         # Archivos JavaScript
│   └── imagenes/                   # Imágenes
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

## APIs Disponibles

- `GET /api/naves` - Obtener lista de naves industriales
- `POST /api/generar-ruta` - Generar ruta optimizada
  - Body: `{"indices": [0, 1, 2, 3, 4]}`
  - Response: Ruta optimizada con coordenadas y condiciones climáticas

## Tecnologías Utilizadas

- **Backend**: Python, Flask, Pandas, NumPy
- **Algoritmo**: Recocido Simulado (implementado en C++)
- **Machine Learning**: Random Forest para predicciones climáticas
- **APIs Externas**: WeatherAPI, OpenRouteService, OSRM
- **Frontend**: HTML5, CSS3, JavaScript, Leaflet.js

## Notas de Desarrollo

- El sistema requiere conexión a internet para las APIs de clima y mapas
- Las predicciones climáticas se realizan en tiempo real
- El algoritmo de optimización puede tomar varios minutos para rutas complejas
- La base de datos de naves industriales está enfocada en el Estado de México

## Próximas Funcionalidades

- [ ] Sistema de autenticación completo
- [ ] Base de datos para guardar rutas
- [ ] Historial de rutas por usuario
- [ ] Exportación de rutas (PDF, Excel)
- [ ] Configuración de parámetros de optimización
