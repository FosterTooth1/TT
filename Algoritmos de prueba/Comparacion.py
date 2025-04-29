import os
import time
import statistics
import csv
import psutil
import ctypes
from ctypes import (
    c_int, c_double, c_char_p, c_char, POINTER, Structure,
    c_float
)
import os
import matplotlib.pyplot as plt

# --------------------------------------------
# Estructuras y clases para Algoritmo Genético
# --------------------------------------------
class ResultadoGenetico(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),
        ("fitness", c_double),
        ("tiempo_ejecucion", c_double),
        ("nombres_ciudades", POINTER(c_char * 50 * 32)),
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
    ]

class AlgoritmoGenetico:
    def __init__(self, ruta_biblioteca):
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        self.biblioteca.ejecutar_algoritmo_genetico.restype = POINTER(ResultadoGenetico)
        self.biblioteca.ejecutar_algoritmo_genetico.argtypes = [
            c_int, c_int, c_int, c_int, c_int,
            c_double, c_double, c_char_p, c_int
        ]
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoGenetico)]

    def ejecutar(self, tamano_poblacion, longitud_genotipo, num_generaciones,
                 num_competidores, m, probabilidad_mutacion, 
                 probabilidad_cruce, nombre_archivo, heuristica):
        try:
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            resultado = self.biblioteca.ejecutar_algoritmo_genetico(
                tamano_poblacion,
                longitud_genotipo,
                num_generaciones,
                num_competidores,
                m,
                probabilidad_mutacion,
                probabilidad_cruce,
                nombre_archivo_bytes,
                heuristica
            )
            if not resultado:
                raise RuntimeError("Error al ejecutar el algoritmo genético")
            
            recorrido = [resultado.contents.recorrido[i] for i in range(resultado.contents.longitud_recorrido)]
            nombres_ciudades = []
            for i in range(resultado.contents.longitud_recorrido):
                nombre_ciudad = bytes(resultado.contents.nombres_ciudades.contents[i]).decode('utf-8').split('\0')[0]
                nombres_ciudades.append(nombre_ciudad)
            
            salida = {
                'recorrido': recorrido,
                'nombres_ciudades': nombres_ciudades,
                'fitness': resultado.contents.fitness,
                'tiempo_ejecucion': resultado.contents.tiempo_ejecucion,
                "fitness_generaciones": [resultado.contents.fitness_generaciones[i] for i in range(num_generaciones)]
            }
            
            self.biblioteca.liberar_resultado(resultado)
            return salida
        except Exception as e:
            raise RuntimeError(f"Error en genético: {str(e)}")

# --------------------------------------------
# Estructuras y clases para PSO
# --------------------------------------------
class ResultadoPSO(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),
        ("fitness", c_double),
        ("tiempo_ejecucion", c_double),
        ("nombres_ciudades", POINTER(c_char * 50)),
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
    ]

class AlgoritmoPSO:
    def __init__(self, ruta_biblioteca):
        self.bibl = ctypes.CDLL(ruta_biblioteca)
        self.bibl.ejecutar_algoritmo_pso.restype = POINTER(ResultadoPSO)
        self.bibl.ejecutar_algoritmo_pso.argtypes = [
            c_int, c_int, c_int, c_double, c_double,
            c_char_p, c_double, c_int, c_int
        ]
        self.bibl.liberar_resultado.argtypes = [POINTER(ResultadoPSO)]

    def ejecutar(self, params):
        ptr = self.bibl.ejecutar_algoritmo_pso(
            params['tamano_poblacion'],
            params['longitud_ruta'],
            params['num_generaciones'],
            params['prob_pbest'],
            params['prob_gbest'],
            params['nombre_archivo'].encode(),
            params['prob_inercia'],
            params['m'],
            params['heuristica'],
        )
        res = ptr.contents

        fg = [res.fitness_generaciones[i] for i in range(params['num_generaciones'])]
        nc = []
        for i in range(res.longitud_recorrido):
            raw = res.nombres_ciudades[i]
            nombre = bytes(raw).split(b'\x00',1)[0].decode()
            nc.append(nombre)

        recorrido = [res.recorrido[i] for i in range(res.longitud_recorrido)]
        self.bibl.liberar_resultado(ptr)

        return {
            'recorrido': recorrido,
            'nombres_ciudades': nc,
            'fitness': res.fitness,
            'tiempo_ejecucion': res.tiempo_ejecucion,
            'fitness_generaciones': fg
        }

# --------------------------------------------
# Estructuras y clases para Recocido Simulado
# --------------------------------------------
class ResultadoRecocido(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),
        ("fitness", c_double),
        ("tiempo_ejecucion", c_double),
        ("nombres_ciudades", POINTER(c_char * 50)),
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
    ]

class AlgoritmoRecocido:
    def __init__(self, ruta_biblioteca):
        self.lib = ctypes.CDLL(ruta_biblioteca)
        self.lib.ejecutar_algoritmo_recocido.restype = POINTER(ResultadoRecocido)
        self.lib.ejecutar_algoritmo_recocido.argtypes = [
            c_int, c_int, c_double, c_double,
            c_int, c_int, c_char_p, c_int
        ]
        self.lib.liberar_resultado_recocido.argtypes = [POINTER(ResultadoRecocido)]

    def ejecutar(self, longitud_ruta, num_generaciones, tasa_enfriamiento, 
                 temperatura_final, max_neighbours, m, nombre_archivo, heuristica):
        ptr = self.lib.ejecutar_algoritmo_recocido(
            longitud_ruta,
            num_generaciones,
            tasa_enfriamiento,
            temperatura_final,
            max_neighbours,
            m,
            nombre_archivo.encode('utf-8'),
            heuristica
        )
        res = ptr.contents

        gens = [res.fitness_generaciones[i] for i in range(num_generaciones)]
        nombres = [res.nombres_ciudades[i].value.decode('utf-8')
                   for i in range(res.longitud_recorrido)]
        ruta = [res.recorrido[i] for i in range(res.longitud_recorrido)]

        self.lib.liberar_resultado_recocido(ptr)
        return {
            "recorrido": ruta,
            "nombres_ciudades": nombres,
            "fitness": res.fitness,
            "tiempo_ejecucion": res.tiempo_ejecucion,
            "fitness_generaciones": gens
        }

# --------------------------------------------
# Estructuras y clases para Búsqueda Tabú
# --------------------------------------------
class ResultadoTabu(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),
        ("fitness", c_double),
        ("tiempo_ejecucion", c_double),
        ("nombres_ciudades", POINTER(c_char * 50)),
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
    ]

class AlgoritmoTabu:
    def __init__(self, ruta_biblioteca):
        self.lib = ctypes.CDLL(ruta_biblioteca)
        self.lib.ejecutar_algoritmo_tabu.restype = POINTER(ResultadoTabu)
        self.lib.ejecutar_algoritmo_tabu.argtypes = [
            c_int, c_int, c_int, c_int,
            c_float, c_float, c_int, c_char_p, c_int
        ]
        self.lib.liberar_resultado_tabu.argtypes = [POINTER(ResultadoTabu)]

    def ejecutar(self, longitud_ruta, tenencia_tabu, num_generaciones,
                 max_neighbours, umbral_est_global, umbral_est_local,
                 m, nombre_archivo, heuristica):
        ptr = self.lib.ejecutar_algoritmo_tabu(
            longitud_ruta,
            tenencia_tabu,
            num_generaciones,
            max_neighbours,
            umbral_est_global,
            umbral_est_local,
            m,
            nombre_archivo.encode('utf-8'),
            heuristica
        )
        res = ptr.contents

        recorrido = [res.recorrido[i] for i in range(res.longitud_recorrido)]
        nombres = [res.nombres_ciudades[i].value.decode('utf-8')
                   for i in range(res.longitud_recorrido)]
        fitness_gens = [res.fitness_generaciones[i]
                        for i in range(num_generaciones)]

        self.lib.liberar_resultado_tabu(ptr)

        return {
            "recorrido": recorrido,
            "nombres_ciudades": nombres,
            "fitness": res.fitness,
            "tiempo_ejecucion": res.tiempo_ejecucion,
            "fitness_generaciones": fitness_gens
        }

S_PARAMS = [
    {   # Algoritmo Genético - Pequeño
        "name": "Genetico_Pequeño",
        "class": AlgoritmoGenetico,
        "library": "genetic_algo.dll" if os.name == 'nt' else "libgenetic_algo.so",
        "params": {
            "tamano_poblacion": 200,
            "longitud_genotipo": 32,
            "num_generaciones": 500,
            "num_competidores": 2,
            "m": 3,
            "probabilidad_mutacion": 0.02,
            "probabilidad_cruce": 0.8,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    },
    {   # PSO
        "name": "PSO_Pequeño",
        "class": AlgoritmoPSO,
        "library": "pso.dll" if os.name == 'nt' else "libpso.so",
        "params": {
            "tamano_poblacion": 200,
            "longitud_ruta": 32,
            "num_generaciones": 500,
            "prob_pbest": 0.35,
            "prob_gbest": 0.7,
            "nombre_archivo": "Distancias_no_head.csv",
            "prob_inercia": 0.3,
            "m": 3,
            "heuristica": 0
        }
    },
    {   # Recocido Simulado
        "name": "Recocido_Pequeño",
        "class": AlgoritmoRecocido,
        "library": "recocido.dll" if os.name == 'nt' else "librecocido.so",
        "params": {
            "longitud_ruta": 32,
            "num_generaciones": 10000,
            "tasa_enfriamiento": 0.95,
            "temperatura_final": 1e-3,
            "max_neighbours": 10,
            "m": 3,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    },
    {   # Búsqueda Tabú
        "name": "Tabu_Pequeño",
        "class": AlgoritmoTabu,
        "library": "tabu.dll" if os.name == 'nt' else "libtabu.so",
        "params": {
            "longitud_ruta": 32,
            "tenencia_tabu": 7,
            "num_generaciones": 10000,
            "max_neighbours": 10,
            "umbral_est_global": 0.1,
            "umbral_est_local": 0.05,
            "m": 3,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    }
]

M_PARAMS = [
    {   # Algoritmo Genético
        "name": "Genetico_Mediano",
        "class": AlgoritmoGenetico,
        "library": "genetic_algo.dll" if os.name == 'nt' else "libgenetic_algo.so",
        "params": {
            "tamano_poblacion": 400,       # 400 ×  500 = 200 000 evals
            "longitud_genotipo": 32,
            "num_generaciones": 500,       
            "num_competidores": 2,
            "m": 3,
            "probabilidad_mutacion": 0.02,
            "probabilidad_cruce": 0.8,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    },
    {   # PSO
        "name": "PSO_Mediano",
        "class": AlgoritmoPSO,
        "library": "pso.dll" if os.name == 'nt' else "libpso.so",
        "params": {
            "tamano_poblacion": 400,       # 400 ×  500 = 200 000 evals
            "longitud_ruta": 32,
            "num_generaciones": 500,       
            "prob_pbest": 0.35,
            "prob_gbest": 0.7,
            "prob_inercia": 0.3,
            "m": 3,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    },
    {   # Recocido Simulado
        "name": "Recocido_Mediano",
        "class": AlgoritmoRecocido,
        "library": "recocido.dll" if os.name == 'nt' else "librecocido.so",
        "params": {
            "longitud_ruta": 32,
            "num_generaciones": 20000,     # 20 000 × 10 = 200 000 evals
            "tasa_enfriamiento": 0.95,
            "temperatura_final": 1e-3,
            "max_neighbours": 10,
            "m": 3,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    },
    {   # Búsqueda Tabú
        "name": "Tabu_Mediano",
        "class": AlgoritmoTabu,
        "library": "tabu.dll" if os.name == 'nt' else "libtabu.so",
        "params": {
            "longitud_ruta": 32,
            "tenencia_tabu": 7,
            "num_generaciones": 20000,     # 20 000 × 10 = 200 000 evals
            "max_neighbours": 10,
            "umbral_est_global": 0.1,
            "umbral_est_local": 0.05,
            "m": 3,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    }
]

B_PARAMS = [
    {   # Algoritmo Genético
        "name": "Genetico_Grande",
        "class": AlgoritmoGenetico,
        "library": "genetic_algo.dll" if os.name == 'nt' else "libgenetic_algo.so",
        "params": {
            "tamano_poblacion": 1000,      # 1000 ×  500 = 500 000 evals
            "longitud_genotipo": 32,
            "num_generaciones": 500,
            "num_competidores": 2,
            "m": 3,
            "probabilidad_mutacion": 0.02,
            "probabilidad_cruce": 0.8,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    },
    {   # PSO
        "name": "PSO_Grande",
        "class": AlgoritmoPSO,
        "library": "pso.dll" if os.name == 'nt' else "libpso.so",
        "params": {
            "tamano_poblacion": 1000,      # 1000 ×  500 = 500 000 evals
            "longitud_ruta": 32,
            "num_generaciones": 500,
            "prob_pbest": 0.35,
            "prob_gbest": 0.7,
            "prob_inercia": 0.3,
            "m": 3,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    },
    {   # Recocido Simulado
        "name": "Recocido_Grande",
        "class": AlgoritmoRecocido,
        "library": "recocido.dll" if os.name == 'nt' else "librecocido.so",
        "params": {
            "longitud_ruta": 32,
            "num_generaciones": 50000,     # 50 000 × 10 = 500 000 evals
            "tasa_enfriamiento": 0.95,
            "temperatura_final": 1e-3,
            "max_neighbours": 10,
            "m": 3,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    },
    {   # Búsqueda Tabú
        "name": "Tabu_Grande",
        "class": AlgoritmoTabu,
        "library": "tabu.dll" if os.name == 'nt' else "libtabu.so",
        "params": {
            "longitud_ruta": 32,
            "tenencia_tabu": 7,
            "num_generaciones": 50000,     # 50 000 × 10 = 500 000 evals
            "max_neighbours": 10,
            "umbral_est_global": 0.1,
            "umbral_est_local": 0.05,
            "m": 3,
            "nombre_archivo": "Distancias_no_head.csv",
            "heuristica": 0
        }
    }
]

# Combinamos todos los conjuntos de parámetros
ALGORITHMS = S_PARAMS + M_PARAMS + B_PARAMS

# ------------------------- Función para realizar las comparaciones -------------------------
def realizar_comparaciones():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Inicializar archivos CSV con nueva estructura
    with open('calidad_solucion.csv', 'w', newline='') as f1, \
         open('tiempo_ejecucion.csv', 'w', newline='') as f2, \
         open('uso_memoria.csv', 'w', newline='') as f3:
        
        writers = [csv.writer(f1), csv.writer(f2), csv.writer(f3)]
        headers = ['Algoritmo', 'Tamaño', 'Mejor', 'Peor', 'Media', 'Desviacion']
        
        for writer in writers:
            writer.writerow(headers)

    for algoritmo in ALGORITHMS:
        print(f"\nEjecutando {algoritmo['name']}...")
        
        fitness_results = []
        time_results = []
        memory_results = []
        
        lib_path = os.path.join(directorio_actual, algoritmo['library'])
        if not os.path.exists(lib_path):
            print(f"¡Biblioteca {lib_path} no encontrada!")
            continue

        for _ in range(30):
            try:
                # Mediciones
                start_time = time.time()
                process = psutil.Process(os.getpid())
                start_mem = process.memory_info().rss
                
                # Ejecución
                instance = algoritmo['class'](lib_path)
                if algoritmo['class'] is AlgoritmoPSO:
                    result = instance.ejecutar(algoritmo['params'])
                else:
                    result = instance.ejecutar(**algoritmo['params'])

                
                # Resultados
                elapsed_time = time.time() - start_time
                end_mem = process.memory_info().rss
                mem_used = (end_mem - start_mem) / 1024**2  # MB
                
                fitness_results.append(result['fitness'])
                time_results.append(elapsed_time)
                memory_results.append(mem_used)
                
            except Exception as e:
                print(f"Error en iteración {_+1}: {str(e)}")
                continue

        # Calcular estadísticas
        def calcular_estadisticas(datos):
            return [
                min(datos) if datos else 0,
                max(datos) if datos else 0,
                statistics.mean(datos) if datos else 0,
                statistics.stdev(datos) if len(datos) > 1 else 0
            ]

        # Extraer nombre base y tamaño CORREGIDO
        if '_' in algoritmo['name']:
            parts = algoritmo['name'].split('_', 1)  # Split solo en el primer _
            nombre_base = parts[0]
            tamaño = parts[1]
        else:
            nombre_base = algoritmo['name']
            tamaño = 'Desconocido'
            print(f"¡Error en formato de nombre! {algoritmo['name']}")
        
        # Escribir resultados
        with open('calidad_solucion.csv', 'a', newline='') as f:
            csv.writer(f).writerow([nombre_base, tamaño] + calcular_estadisticas(fitness_results))
        
        with open('tiempo_ejecucion.csv', 'a', newline='') as f:
            csv.writer(f).writerow([nombre_base, tamaño] + calcular_estadisticas(time_results))
        
        with open('uso_memoria.csv', 'a', newline='') as f:
            csv.writer(f).writerow([nombre_base, tamaño] + calcular_estadisticas(memory_results))

if __name__ == "__main__":
    realizar_comparaciones()
    print("\n¡Comparaciones completadas! Ver archivos CSV.")