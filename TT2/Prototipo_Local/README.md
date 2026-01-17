# LogistiClima - Prototipo local

## Descripción
Algoritmo de optimización de rutas logísticas que integra predicciones climáticas para naves industriales usando el algoritmo de Recocido Simulado. Este prototipo constituye el núcleo computacional del sistema LogistiClima, responsable del procesamiento de datos, generación de matrices de distancias y optimización de rutas.

## Características
- **Optimización de rutas** mediante algoritmo de recocido simulado implementado en C
- **Predicciones climáticas** en tiempo real con modelo de machine learning
- **Procesamiento de datos** automatizado para limpieza y preparación
- **Generación de matrices de distancias** utilizando APIs de ruteo real
- **Penalización dinámica** de rutas según condiciones climáticas
- **API REST** para integración con sistemas de frontend

## Instalación

### Requisitos
- Python 3.8+
- GCC/Clang (para compilar las librerías C)
- Dependencias Python listadas en `requirements.txt`
- Archivos de configuración: `.env`
- APIs Keys: WeatherAPI, OpenRouteService

### Pasos de instalación

1. **Clonar el repositorio principal**:
```bash
git clone https://github.com/FosterTooth1/TT.git
cd TT2\Prototipo_Local
```

2. **Crear ambiente virtual:**
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno:**
Crear archivo `.env` en la raíz del proyecto:
```
WEATHER_API_KEY=tu_api_key_weatherapi
OPENROUTESERVICE_API_KEY=tu_api_key_openrouteservice
```

5. **Compilar librerías C:**
```bash
# En la carpeta src/c/
# Windows:
gcc -shared -o ../../lib/recocido.dll -fPIC Main.c Funciones.c

# Linux:
gcc -shared -o ../../lib/librecocido.so -fPIC Main.c Funciones.c
```

6. **Verificar archivos necesarios:**
   - `data/processed/Naves_Industriales.csv` - Base de datos de naves
   - `data/processed/Matriz_Distancias_Carretera.csv` - Matriz de distancias
   - `models/prediccion_clima.pkl` - Modelo ML entrenado
   - `lib/recocido.dll` (Windows) o `lib/librecocido.so` (Linux)

## Uso

### Flujo principal

1. **Preprocesamiento de datos** (si hay nuevos datos)
```bash
jupyter notebook notebooks/Preprocesamiento_naves.ipynb
jupyter notebook notebooks/Preprocesamiento_municipios.ipynb
```

2. **Generar matriz de distancias** (si hay nuevas naves)
```bash
python src/python/Crear_Matriz_Distancias.py
```

3. **Ejecutar optimización de rutas**
```bash
python src/python/Main.py
```
   - El programa solicita seleccionar naves industriales
   - Realiza predicciones climáticas (opcional)
   - Aplica penalizaciones por clima (opcional)
   - Genera ruta optimizada y la guarda en `output/ruta_Ejemplo.json`

4. **Servir API REST**
```bash
uvicorn src.api.API_Logisticlima:app --reload --port 8000
```

### Funcionalidades

**Selección de naves:**
- Elige índices específicos o todas las naves (-1)
- Formato: `0,1,3,5` o `-1` para todas

**Predicciones climáticas:**
- Consulta en tiempo real condiciones actuales
- Integración con WeatherAPI
- Mapeo de condiciones a penalizaciones

**Optimización:**
- Considera distancias reales por carretera
- Aplica factores de penalización por clima
- Genera ruta optimizada con estadísticas

## Estructura del proyecto

```
Prototipo_1/
├── src/
│   ├── c/                          # Código C del algoritmo
│   │   ├── Main.c                 # Implementación del recocido simulado
│   │   ├── Funciones.c            # Funciones auxiliares
│   │   └── Bibliotecas.h          # Headers y definiciones
│   ├── python/                     # Scripts Python principales
│   │   ├── Main.py                # Orquestador principal
│   │   └── Crear_Matriz_Distancias.py  # Generador de matrices
│   └── api/
│       └── API_Logisticlima.py    # Servicio API REST
├── data/
│   ├── raw/                        # Datos sin procesar
│   │   ├── CSV_Sucio_Naves.csv
│   │   └── Municipios_Sucios.csv
│   └── processed/                  # Datos procesados y limpios
│       ├── Naves_Industriales.csv
│       ├── Matriz_Distancias_Carretera.csv
│       ├── Municipios.csv
│       └── Naves_Industriales_Con_Predicciones.csv
├── notebooks/                      # Jupyter Notebooks para el preprocesamiento de datos
│   ├── Preprocesamiento_naves.ipynb
│   └── Preprocesamiento_municipios.ipynb
├── lib/                            # Librerías compiladas
│   ├── recocido.dll               # Windows
│   └── librecocido.so             # Linux
├── models/                         # Modelos ML entrenados
│   └── prediccion_clima.pkl       # Random Forest para predicciones
├── database/                       # Esquema de base de datos
│   └── BDD.sql
├── output/                         # Resultados y salidas
│   └── ruta_Ejemplo.json          # Rutas optimizadas generadas
├── requirements.txt                # Dependencias Python
├── .env                           # Variables de entorno
└── README.md                      # Este archivo
```

## Componentes principales

### Recocido simulado (src/c/)
Algoritmo de optimización metaheurístico implementado en C para máximo rendimiento.
- **Parámetros ajustables**: temperatura, tasa de enfriamiento, número de generaciones
- **Heurística de abruptos**: heuristíca "k-neighbours"
- **Métrica**: Distancia total del recorrido

### Predicción climática (models/)
Modelo random forest entrenado para clasificar 10 categorías de condiciones climáticas.
- Input: Hora, temperatura, punto de rocío, humedad, dirección/velocidad del viento
- Output: Categoría de clima con factor de penalización (1.0 a 1.9)

### Generador de matrices (src/python/Crear_Matriz_Distancias.py)
Construye matriz de distancias reales por carretera usando API OpenRouteService.
- Procesamiento por lotes para evitar límites de API
- Distancias en kilómetros
- Cálculo simétrico de la matriz

### Orquestador principal (src/python/Main.py)
Integra todos los componentes en un flujo unificado.
- Carga datos y modelos
- Permite selección interactiva de naves
- Aplica predicciones y penalizaciones
- Ejecuta optimización
- Exporta resultados en JSON

## Tecnologías utilizadas

- **Backend**: Python 3.8+, Pandas, NumPy, Scikit-learn
- **Optimización**: Algoritmo de recocido simulado en C
- **Machine learning**: Random forest (predicciones climáticas)
- **APIs Externas**: 
  - WeatherAPI: Datos climáticos en tiempo real
  - OpenRouteService: Matrices de distancias reales
- **Serialización**: Joblib (modelos), JSON (resultados)
- **Notebooks**: Jupyter para análisis exploratorio y procesamiento de datos

## Notas de desarrollo

- Las predicciones climáticas requieren conexión a internet
- La generación de matrices es costosa en API
- El algoritmo de recocido puede optimizarse con tuning de parámetros
- Los datos están enfocados en el Estado de México
- Las librerías compiladas deben coincidir con el sistema operativo
- Las rutas relativas funcionan desde cualquier ubicación de ejecución

## Parámetros de optimización

Edita estos valores en `Main.py` para ajustar el comportamiento del algoritmo:

```python
params = {
    'longitud_ruta': num_naves,        # Número de destinos
    'num_generaciones': 800,           # Iteraciones del algoritmo
    'tasa_enfriamiento': 0.99,         # Velocidad de enfriamiento
    'temperatura_final': 0.001,        # Criterio de parada
    'max_neighbours': num_naves * 10,  # Vecinos a explorar
    'm': 3,                            # Ciudades cercanas en heurística
    'nombre_archivo': ruta_matriz,     # Archivo de matriz matriz de distancias
    'heuristica': 0                    # Aplica heuristica (bool)
}
```

## Troubleshooting

**Error: No se encuentra la librería compilada**
- Verificar que `lib/recocido.dll` (Windows) o `lib/librecocido.so` (Linux) exista
- Recompilar si es necesario

**Error: API Key inválida**
- Verificar que `.env` esté en la raíz del proyecto
- Verificar credenciales en variables de entorno

**Error: Archivo de matriz no encontrado**
- Ejecutar primero `Crear_Matriz_Distancias.py`
- Verificar que archivo esté en `data/processed/`

