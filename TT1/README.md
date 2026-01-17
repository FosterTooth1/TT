# TT1 - Desarrollo, implementación y comparación de algoritmos bioinspirados, modelos de ML y de la API metereológica

## Descripción general
Este modulo del proyecto incluye:
- Implementaciones en C y wrappers en Python de cuatro algoritmos bioinspirados: **Algoritmo genético (GA)**, **Optimización por enjambre de partículas (PSO)**, **Recocido simulado (simulated annealing)** y **Búsqueda tabú**.
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
│       ├── Dataset_Final_CE.csv      # Dataset de entrenamiento para modelos de ML
│       └── Matriz_Distancias.csv
├── lib/                               # Librerías compiladas (.dll/.so)
├── notebooks/
│   ├── .env
│   ├── .gitignore
│   ├── TT_solicitud_clima_actual.ipynb   # Evaluación y elección de API meteorológica
│   └── modelos_predictivos/              # Comparativa de modelos de predicción
│       ├── TT_prueba_Arbol.ipynb         # Genera: arbol_decision_model.pkl
│       ├── TT_prueba_LSTM.ipynb          # Genera: lstm_model.keras, lstm_encoders.pkl
│       ├── TT_prueba_Random_Forest.ipynb # Genera: random_forest_model.pkl
│       ├── TT_prueba_RNN.ipynb           # Genera: rnn_model.keras, rnn_encoders.pkl
│       └── TT_prueba_SVM.ipynb           # Genera: svm_model.pkl
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
        ├── recocido/ (Bibliotecas.h, Funciones.c, Funciones.o, Main.c, Main.o, Main.py, recocido.dll, librecocido.so)
        └── tabu/     (Bibliotecas.h, Funciones.c, Funciones.o, Main.c, Main.o, Main.py, tabu.dll, libtabu.so)
                         
```

## Uso

### Compilar librerías C
```bash
cd src/c/genetico
gcc -shared -o ../../../lib/genetic_algo.dll -fPIC Main.c Funciones.c
gcc -shared -o ../../../lib/libgenetic_algo.so -fPIC Main.c Funciones.c
```

### Ejecutar algoritmo individual
```bash
cd src/python/genetico
python Main.py
```

### Ejecutar comparación de algoritmos
```bash
cd src/python/comparacion
python Comparacion.py
```

### Visualizar resultados
```bash
cd src/python/comparacion
python Visualizar_resultados.py
```

### Ejecutar notebooks de modelos predictivos
Los notebooks en `notebooks/modelos_predictivos/` están configurados para ejecutarse localmente:

1. **Fuente de datos**: Utilizan el dataset `data/processed/Dataset_Final_CE.csv`
2. **Modelos generados**: Cada notebook genera archivos `.pkl` (o `.keras` para redes neuronales) con el modelo entrenado y los encoders necesarios
3. **Ejecución**: Abrir el notebook en VS Code o Jupyter y ejecutar todas las celdas

**Archivos generados por cada modelo:**
| Notebook | Modelo | Encoders/Scaler |
|----------|--------|-----------------|
| TT_prueba_Arbol.ipynb | `arbol_decision_model.pkl` | `arbol_decision_encoders.pkl` |
| TT_prueba_Random_Forest.ipynb | `random_forest_model.pkl` | `random_forest_encoders.pkl` |
| TT_prueba_SVM.ipynb | `svm_model.pkl` | `svm_encoders.pkl` |
| TT_prueba_LSTM.ipynb | `lstm_model.keras` | `lstm_encoders.pkl`, `lstm_scaler.pkl` |
| TT_prueba_RNN.ipynb | `rnn_model.keras` | `rnn_encoders.pkl`, `rnn_scaler.pkl` |

## Dependencias python
- ctypes
- matplotlib
- pandas
- chardet
- scikit-learn
- tensorflow
- seaborn
- joblib
