# TT1 - Desarrollo, implementación y comparación de Algoritmos Bioinspirados, Modelos de ML y de la API metereológica

## Descripción General
Este modulo del proyecto incluye:
- Implementaciones en C y wrappers en Python de cuatro algoritmos bioinspirados: **Algoritmo Genético (GA)**, **Optimización por Enjambre de Partículas (PSO)**, **Recocido Simulado (Simulated Annealing)** y **Búsqueda Tabú**.
- Experimentación y selección de la API meteorológica a consumir, documentada en el notebook **TT_solicitud_clima_actual.ipynb**.
- Comparación de distintas implementaciones de modelos de ML para predicción meteorológica, ubicada en `notebooks/modelos_predictivos/`.

Los scripts Python usan rutas relativas, permitiendo ejecución desde cualquier directorio, y los wrappers cargan las bibliotecas C mediante `ctypes`.

## Estructura del modulo

```
TT1/
├── README.md
├── requirements.txt
├── data/
│   └── processed/
│       └── Matriz_Distancias.csv
├── lib/                               # Librerías compiladas (.dll/.so)
├── notebooks/
│   ├── .env
│   ├── .gitignore
│   ├── TT_solicitud_clima_actual.ipynb   # Evaluación y elección de API meteorológica
│   └── modelos_predictivos/              # Comparativa de modelos de predicción
│       ├── TT_prueba_Arbol.ipynb
│       ├── TT_prueba_LSTM.ipynb
│       ├── TT_prueba_Random_Forest.ipynb
│       ├── TT_prueba_RNN.ipynb
│       └── TT_prueba_SVM.ipynb
├── src/
    ├── c/
    │   ├── genetico/ (Bibliotecas.h, Funciones.c, Main.c, genetico.exe, a.out)
    │   ├── pso/      (Bibliotecas.h, Funciones.c, Main.c, pso.exe, a.exe)
    │   ├── recocido/ (Bibliotecas.h, Funciones.c, Main.c, recocido.exe, a.exe)
    │   └── tabu/     (Bibliotecas.h, Funciones.c, Main.c, tabu.exe, a.exe)
    └── python/                           # Wrappers y benchmarking
        ├── comparacion/
        │   ├── Comparacion.py
        │   ├── Visualizar_resultados.py
        │   ├── reporte_comparacion.html
        │   ├── resultados_calidad/
        │   ├── resultados_estabilidad/
        │   ├── resultados_tiempo/
        │   └── __pycache__/
        ├── genetico/ (Bibliotecas.h, Funciones.c, Main.c, Main.py, genetic_algo.dll, libgenetic_algo.so)
        ├── pso/      (Bibliotecas.h, Funciones.c, Main.c, Main.py, pso.dll, libpso.so)
        ├── recocido/ (Bibliotecas.h, Funciones.c, Funciones.o, Main.c, Main.o, Main.py,    recocido.dll, librecocido.so)
        └── tabu/     (Bibliotecas.h, Funciones.c, Funciones.o, Main.c, Main.o, Main.py, tabu.dll, libtabu.so)
                         
```

## Uso

### Compilar Librerías C
```bash
cd src/c/genetico
gcc -shared -o ../../../lib/genetic_algo.dll -fPIC Main.c Funciones.c
gcc -shared -o ../../../lib/libgenetic_algo.so -fPIC Main.c Funciones.c
```

### Ejecutar Algoritmo Individual
```bash
cd src/python/genetico
python Main.py
```

### Ejecutar Comparación de Algoritmos
```bash
cd src/python/comparacion
python Comparacion.py
```

### Visualizar Resultados
```bash
cd src/python/comparacion
python Visualizar_resultados.py
```

## Dependencias Python
- ctypes
- matplotlib
- pandas
- chardet
