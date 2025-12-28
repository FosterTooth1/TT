import os
import time
import statistics
import csv
import ctypes
from ctypes import c_int, c_double, c_char_p, c_char, POINTER, Structure, c_float, cast
import os
import gc

# Clases y estructuras para el algoritmo genético
class ResultadoGenetico(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),         # Puntero al arreglo de la mejor ruta
        ("fitness", c_double),                # Fitness del mejor individuo
        ("tiempo_ejecucion", c_double),       # Tiempo de ejecución del algoritmo
        ("nombres_ciudades", POINTER(c_char * 50 * 32)),  # Puntero a los nombres de las ciudades
        ("longitud_recorrido", c_int),         # Longitud de la ruta
        ("fitness_generaciones", POINTER(c_double)),
    ]

# Clase para la biblioteca compartida del algoritmo genético
class AlgoritmoGenetico:
    def __init__(self, ruta_biblioteca):
        # Cargamos la biblioteca
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        
        # Configuramos el tipo de retorno de la función "ejecutar_algoritmo_genetico"
        self.biblioteca.ejecutar_algoritmo_genetico.restype = POINTER(ResultadoGenetico)
        
        # Especificamos los tipos de argumentos que espera "ejecutar_algoritmo_genetico"
        self.biblioteca.ejecutar_algoritmo_genetico.argtypes = [
            c_int,      # tamano_poblacion
            c_int,      # longitud_genotipo
            c_int,      # num_generaciones
            c_int,      # num_competidores
            c_int,      # m parametro de heurística
            c_double,   # probabilidad_mutacion
            c_double,   # probabilidad_cruce
            c_char_p,   # nombre_archivo (ruta al archivo con matriz de distancias)
            c_int       # Heuristica (0 o 1) 
        ]
        
        # Configuramos los argumentos de la función para liberar resultados
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoGenetico)]

    def ejecutar(self, tamano_poblacion, longitud_genotipo, num_generaciones,
                 num_competidores, m, probabilidad_mutacion, 
                 probabilidad_cruce, nombre_archivo, heuristica):
        try:
            # Convertimos el nombre del archivo a una cadena de bytes
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            
            # Llamamos a la función "ejecutar_algoritmo_genetico" de la biblioteca C
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
            
            # Verificamos si la función devolvió un resultado válido
            if not resultado:
                raise RuntimeError("Error al ejecutar el algoritmo genético")
            
            # Convertimos el recorrido (índices de las ciudades) a una lista de Python
            recorrido = [resultado.contents.recorrido[i] for i in range(resultado.contents.longitud_recorrido)]
            
            # Convertimos los nombres de las ciudades a una lista de Python
            nombres_ciudades = []
            for i in range(resultado.contents.longitud_recorrido):
                # Cada ciudad es un array de caracteres en C que convertimos a cadena de Python
                nombre_ciudad = bytes(resultado.contents.nombres_ciudades.contents[i]).decode('utf-8')
                nombre_ciudad = nombre_ciudad.split('\0')[0]  # Eliminamos los caracteres nulos
                nombres_ciudades.append(nombre_ciudad)
                
            # Creamos un diccionario con los resultados
            salida = {
                'recorrido': recorrido,                 # Ruta como lista de índices
                'nombres_ciudades': nombres_ciudades,   # Lista de nombres de las ciudades
                'fitness': resultado.contents.fitness,  # Fitness del mejor individuo
                'tiempo_ejecucion': resultado.contents.tiempo_ejecucion,  # Tiempo de ejecución
                "fitness_generaciones": [resultado.contents.fitness_generaciones[i] for i in range(num_generaciones)] # Evolución del fitness
            }
            
            # Liberamos la memoria reservada por la biblioteca C
            self.biblioteca.liberar_resultado(resultado)
            
            return salida  # Devolvemos los resultados como un diccionario
            
        except Exception as e:
            raise RuntimeError(f"Error al ejecutar el algoritmo genético: {str(e)}")

# Clases y estructuras para el algoritmo PSO
class ResultadoPSO(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),          # Puntero al mejor recorrido
        ("fitness", c_double),                 # Valor de fitness
        ("tiempo_ejecucion", c_double),        # Tiempo de ejecución
        ("nombres_ciudades", POINTER(c_char * 50 * 32)),  # Array de nombres (32 ciudades)
        ("longitud_recorrido", c_int),          # Longitud del recorrido
        ("fitness_generaciones", POINTER(c_double))  # Evolución del fitness
    ]

# Clase para la biblioteca compartida del algoritmo PSO
class AlgoritmoPSO:
    def __init__(self, ruta_biblioteca):
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        
        # Configurar tipos de la función ejecutar_algoritmo_pso
        self.biblioteca.ejecutar_algoritmo_pso.restype = POINTER(ResultadoPSO)
        self.biblioteca.ejecutar_algoritmo_pso.argtypes = [
            c_int,      # tamano_poblacion
            c_int,      # longitud_ruta
            c_int,      # num_generaciones
            c_double,   # prob_pbest
            c_double,   # prob_gbest
            c_char_p,   # nombre_archivo
            c_double,   # prob_inercia
            c_int,      # m
            c_int       # heuristica
        ]
        
        # Configurar función de liberación
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoPSO)]

    def ejecutar(self, tamano_poblacion, longitud_ruta, num_generaciones,
                 prob_pbest, prob_gbest, nombre_archivo, 
                 prob_inercia, m, heuristica):
        try:
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            
            # Llamar a la función C
            resultado_ptr = self.biblioteca.ejecutar_algoritmo_pso(
                tamano_poblacion,
                longitud_ruta,
                num_generaciones,
                c_double(prob_pbest),
                c_double(prob_gbest),
                nombre_archivo_bytes,
                c_double(prob_inercia),
                m,
                heuristica
            )
            
            if not resultado_ptr:
                raise RuntimeError("Error en la ejecución del PSO")

            resultado = resultado_ptr.contents
            
            # Copiar datos a estructuras Python
            recorrido = [resultado.recorrido[i] for i in range(resultado.longitud_recorrido)]
            
            nombres_ciudades = []
            ciudades_array = cast(resultado.nombres_ciudades, POINTER(c_char * 50 * 32))  # Cast explícito
            for i in range(resultado.longitud_recorrido):
                nombre_bytes = bytes(ciudades_array.contents[i])
                nombre = nombre_bytes.decode('utf-8').split('\x00', 1)[0]
                nombres_ciudades.append(nombre)
            
            # Copiar histórico de fitness
            fitness_hist = [resultado.fitness_generaciones[i] for i in range(num_generaciones)]
            
            salida = {
                'recorrido': recorrido,
                'nombres_ciudades': nombres_ciudades,
                'fitness': resultado.fitness,
                'tiempo_ejecucion': resultado.tiempo_ejecucion,
                'fitness_generaciones': fitness_hist
            }
            
            # Liberar memoria C
            self.biblioteca.liberar_resultado(resultado_ptr)
            
            return salida
            
        except Exception as e:
            raise RuntimeError(f"Error en PSO: {str(e)}")

# Clases y estructuras para el Recocido Simulado
class ResultadoRecocido(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),
        ("fitness", c_double),
        ("tiempo_ejecucion", c_double),
        ("nombres_ciudades", POINTER(c_char * 50 * 32)),
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
    ]

# Clase para la biblioteca compartida del Recocido Simulado
class AlgoritmoRecocido:
    def __init__(self, ruta_biblioteca):
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        self.biblioteca.ejecutar_algoritmo_recocido.restype = POINTER(ResultadoRecocido)
        self.biblioteca.ejecutar_algoritmo_recocido.argtypes = [
            c_int,      # longitud_ruta
            c_int,      # num_generaciones
            c_double,   # tasa_enfriamiento
            c_double,   # temperatura_final
            c_int,      # max_neighbours
            c_int,      # m
            c_char_p,   # nombre_archivo
            c_int       # heuristica
        ]
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoRecocido)]  # Mismo nombre de función

    def ejecutar(self, longitud_ruta, num_generaciones, tasa_enfriamiento,
               temperatura_final, max_neighbours, m, nombre_archivo, heuristica):
        try:
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            
            resultado_ptr = self.biblioteca.ejecutar_algoritmo_recocido(
                c_int(longitud_ruta),
                c_int(num_generaciones),
                c_double(tasa_enfriamiento),
                c_double(temperatura_final),
                c_int(max_neighbours),
                c_int(m),
                nombre_archivo_bytes,
                c_int(heuristica)
            )
            
            if not resultado_ptr:
                raise RuntimeError("Error en ejecución del Recocido")
            
            resultado = resultado_ptr.contents
            
            # Copia de datos
            recorrido = [resultado.recorrido[i] for i in range(resultado.longitud_recorrido)]
            
            nombres_ciudades = []
            ciudades_array = cast(resultado.nombres_ciudades, POINTER(c_char * 50 * 32))
            for i in range(resultado.longitud_recorrido):
                nombre_bytes = bytes(ciudades_array.contents[i])
                nombre = nombre_bytes.decode('utf-8').split('\x00', 1)[0]
                nombres_ciudades.append(nombre)
            
            fitness_hist = [resultado.fitness_generaciones[i] for i in range(num_generaciones)]
            
            salida = {
                'recorrido': recorrido,
                'nombres_ciudades': nombres_ciudades,
                'fitness': resultado.fitness,
                'tiempo_ejecucion': resultado.tiempo_ejecucion,
                'fitness_generaciones': fitness_hist
            }
            
            self.biblioteca.liberar_resultado(resultado_ptr)
            
            return salida
            
        except Exception as e:
            raise RuntimeError(f"Error en Recocido Simulado: {str(e)}")

# Clases y estructuras para la Búsqueda Tabú
class ResultadoTabu(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),
        ("fitness", c_double),
        ("tiempo_ejecucion", c_double),
        ("nombres_ciudades", POINTER(c_char * 50 * 32)),  
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
    ]

class AlgoritmoTabu:
    def __init__(self, ruta_biblioteca):
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        self.biblioteca.ejecutar_algoritmo_tabu.restype = POINTER(ResultadoTabu)
        self.biblioteca.ejecutar_algoritmo_tabu.argtypes = [
            c_int,      # longitud_ruta
            c_int,      # tenencia_tabu
            c_int,      # num_generaciones
            c_int,      # max_neighbours
            c_float,    # umbral_est_global
            c_float,    # umbral_est_local
            c_int,      # m
            c_char_p,   # nombre_archivo
            c_int       # heuristica
        ]
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoTabu)] 

    def ejecutar(self, longitud_ruta, tenencia_tabu, num_generaciones,
               max_neighbours, umbral_est_global, umbral_est_local,
               m, nombre_archivo, heuristica):
        try:
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            
            resultado_ptr = self.biblioteca.ejecutar_algoritmo_tabu(
                c_int(longitud_ruta),
                c_int(tenencia_tabu),
                c_int(num_generaciones),
                c_int(max_neighbours),
                c_float(umbral_est_global),
                c_float(umbral_est_local),
                c_int(m),
                nombre_archivo_bytes,
                c_int(heuristica)
            )
            
            if not resultado_ptr:
                raise RuntimeError("Error en ejecución de Tabu Search")
            
            resultado = resultado_ptr.contents
            
            # Extracción de datos
            recorrido = [resultado.recorrido[i] for i in range(resultado.longitud_recorrido)]
            
            nombres_ciudades = []
            ciudades_array = cast(resultado.nombres_ciudades, POINTER(c_char * 50 * 32))
            for i in range(resultado.longitud_recorrido):
                nombre_bytes = bytes(ciudades_array.contents[i])
                nombre = nombre_bytes.decode('utf-8').split('\x00', 1)[0]
                nombres_ciudades.append(nombre)
            
            fitness_hist = [resultado.fitness_generaciones[i] for i in range(num_generaciones)]
            
            salida = {
                'recorrido': recorrido,
                'nombres_ciudades': nombres_ciudades,
                'fitness': resultado.fitness,
                'tiempo_ejecucion': resultado.tiempo_ejecucion,
                'fitness_generaciones': fitness_hist
            }
            
            self.biblioteca.liberar_resultado(resultado_ptr)
            
            return salida
            
        except Exception as e:
            raise RuntimeError(f"Error en Tabu Search: {str(e)}")

# Helper para obtener ruta de matriz
def get_matriz_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", "data", "processed", "Matriz_Distancias.csv")

# Helper para obtener ruta de librerías
def get_lib_path(lib_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", "lib", lib_name)

# Parâmetros para los diferentes tamaños de ejecución
# Parametros para 100,000 evaluaciones
S_PARAMS = [
    {   # Algoritmo Genético
        "name": "Genetico_100,000 evaluaciones",
        "class": AlgoritmoGenetico,
        "library": get_lib_path("genetic_algo.dll") if os.name == 'nt' else get_lib_path("libgenetic_algo.so"),
        "params": {
            "tamano_poblacion": 200,
            "longitud_genotipo": 32,
            "num_generaciones": 500,
            "num_competidores": 2,
            "m": 3,
            "probabilidad_mutacion": 0.02,
            "probabilidad_cruce": 0.8,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    },
    {   # PSO
        "name": "PSO_100,000 evaluaciones",
        "class": AlgoritmoPSO,
        "library": get_lib_path("pso.dll" if os.name == 'nt' else "libpso.so"),
        "params": {
            "tamano_poblacion": 200,
            "longitud_ruta": 32,
            "num_generaciones": 500,
            "prob_pbest": 0.35,
            "prob_gbest": 0.7,
            "nombre_archivo": get_matriz_path(),
            "prob_inercia": 0.3,
            "m": 3,
            "heuristica": 0
        }
    },
    {   # Recocido Simulado
        "name": "Recocido_100,000 evaluaciones",
        "class": AlgoritmoRecocido,
        "library": get_lib_path("recocido.dll" if os.name == 'nt' else "librecocido.so"),
        "params": {
            "longitud_ruta": 32,
            "num_generaciones": 10000,
            "tasa_enfriamiento": 0.95,
            "temperatura_final": 1e-3,
            "max_neighbours": 10,
            "m": 3,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    },
    {   # Búsqueda Tabú
        "name": "Tabu_100,000 evaluaciones",
        "class": AlgoritmoTabu,
        "library": get_lib_path("tabu.dll" if os.name == 'nt' else "libtabu.so"),
        "params": {
            "longitud_ruta": 32,
            "tenencia_tabu": 7,
            "num_generaciones": 10000,
            "max_neighbours": 10,
            "umbral_est_global": 0.1,
            "umbral_est_local": 0.05,
            "m": 3,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    }
]

# Parametros para 500,000 evaluaciones
M_PARAMS = [
    {   # Algoritmo Genético
        "name": "Genetico_500,000 evaluaciones",
        "class": AlgoritmoGenetico,
        "library": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", "lib", "genetic_algo.dll") if os.name == 'nt' else os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", "lib", "libgenetic_algo.so"),
        "params": {
            "tamano_poblacion": 500,     
            "longitud_genotipo": 32,
            "num_generaciones": 1000,       
            "num_competidores": 2,
            "m": 3,
            "probabilidad_mutacion": 0.02,
            "probabilidad_cruce": 0.8,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    },
    {   # PSO
        "name": "PSO_500,000 evaluaciones",
        "class": AlgoritmoPSO,
        "library": get_lib_path("pso.dll" if os.name == 'nt' else "libpso.so"),
        "params": {
            "tamano_poblacion": 500,      
            "longitud_ruta": 32,
            "num_generaciones": 1000,       
            "prob_pbest": 0.35,
            "prob_gbest": 0.7,
            "prob_inercia": 0.3,
            "m": 3,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    },
    {   # Recocido Simulado
        "name": "Recocido_500,000 evaluaciones",
        "class": AlgoritmoRecocido,
        "library": get_lib_path("recocido.dll" if os.name == 'nt' else "librecocido.so"),
        "params": {
            "longitud_ruta": 32,
            "num_generaciones": 20000,   
            "tasa_enfriamiento": 0.95,
            "temperatura_final": 1e-3,
            "max_neighbours": 25,
            "m": 3,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    },
    {   # Búsqueda Tabú
        "name": "Tabu_500,000 evaluaciones",
        "class": AlgoritmoTabu,
        "library": get_lib_path("tabu.dll" if os.name == 'nt' else "libtabu.so"),
        "params": {
            "longitud_ruta": 32,
            "tenencia_tabu": 7,
            "num_generaciones": 20000,
            "max_neighbours": 25,
            "umbral_est_global": 0.1,
            "umbral_est_local": 0.05,
            "m": 3,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    }
]

# Parametros para 2,000,000 evaluaciones
B_PARAMS = [
    {   # Algoritmo Genético
        "name": "Genetico_2,000,000 evaluaciones",
        "class": AlgoritmoGenetico,
        "library": get_lib_path("genetic_algo.dll" if os.name == 'nt' else "libgenetic_algo.so"),
        "params": {
            "tamano_poblacion": 1000,   
            "longitud_genotipo": 32,
            "num_generaciones": 2000,
            "num_competidores": 2,
            "m": 3,
            "probabilidad_mutacion": 0.02,
            "probabilidad_cruce": 0.8,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    },
    {   # PSO
        "name": "PSO_2,000,000 evaluaciones",
        "class": AlgoritmoPSO,
        "library": get_lib_path("pso.dll" if os.name == 'nt' else "libpso.so"),
        "params": {
            "tamano_poblacion": 1000,      
            "longitud_ruta": 32,
            "num_generaciones": 2000,
            "prob_pbest": 0.35,
            "prob_gbest": 0.7,
            "prob_inercia": 0.3,
            "m": 3,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    },
    {   # Recocido Simulado
        "name": "Recocido_2,000,000 evaluaciones",
        "class": AlgoritmoRecocido,
        "library": get_lib_path("recocido.dll" if os.name == 'nt' else "librecocido.so"),
        "params": {
            "longitud_ruta": 32,
            "num_generaciones": 80000,   
            "tasa_enfriamiento": 0.95,
            "temperatura_final": 1e-3,
            "max_neighbours": 25,
            "m": 3,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    },
    {   # Búsqueda Tabú
        "name": "Tabu_2,000,000 evaluaciones",
        "class": AlgoritmoTabu,
        "library": get_lib_path("tabu.dll" if os.name == 'nt' else "libtabu.so"),
        "params": {
            "longitud_ruta": 32,
            "tenencia_tabu": 7,
            "num_generaciones": 40000,   
            "max_neighbours": 50,
            "umbral_est_global": 0.1,
            "umbral_est_local": 0.05,
            "m": 3,
            "nombre_archivo": get_matriz_path(),
            "heuristica": 0
        }
    }
]

# Combinamos todos los conjuntos de parámetros
ALGORITHMS = S_PARAMS + M_PARAMS + B_PARAMS

# Función principal para realizar las comparaciones
def realizar_comparaciones():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    
    carpetas = ['resultados_calidad', 'resultados_tiempo', 'resultados_estabilidad']
    for carpeta in carpetas:
        if not os.path.exists(os.path.join(directorio_actual, carpeta)):
            os.makedirs(os.path.join(directorio_actual, carpeta))
            
    path_calidad = os.path.join(directorio_actual, 'resultados_calidad', 'calidad_solucion.csv')
    path_tiempo = os.path.join(directorio_actual, 'resultados_tiempo', 'tiempo_ejecucion.csv')
    path_estabilidad = os.path.join(directorio_actual, 'resultados_estabilidad', 'estabilidad.csv')
    
    # Inicializar archivos CSV con cabeceras
    with open(path_calidad, 'w', newline='') as f1, \
         open(path_tiempo, 'w', newline='') as f2, \
         open(path_estabilidad, 'w', newline='') as f3:
        
        # Cabeceras actualizadas
        csv.writer(f1).writerow(['Algoritmo', 'Tamaño', 'Mejor', 'Peor', 'Media', 'Desviacion'])
        csv.writer(f2).writerow(['Algoritmo', 'Tamaño', 'Mejor', 'Peor', 'Media', 'Desviacion'])
        csv.writer(f3).writerow(['Algoritmo', 'Tamaño', 'CoeficienteVariacion'])

    for algoritmo in ALGORITHMS:
        print(f"\nEjecutando {algoritmo['name']}...")
        
        fitness_results = []
        time_results = []
        
        lib_path = os.path.join(directorio_actual, algoritmo['library'])
        if not os.path.exists(lib_path):
            print(f"Biblioteca {lib_path} no encontrada")
            continue

        for _ in range(30):
            try:
                # Mediciones de tiempo
                start_time = time.time()
                
                # Ejecución
                instance = algoritmo['class'](lib_path)
                result = instance.ejecutar(**algoritmo['params'])
                
                elapsed_time = time.time() - start_time
                
                # Almacenar resultados
                fitness_results.append(result['fitness'])
                time_results.append(elapsed_time)
                
            except Exception as e:
                print(f"Error en iteración {_+1}: {str(e)}")
                continue

        # Cálculo de estadísticas
        def calcular_estadisticas(datos):
            if not datos:
                return [0, 0, 0, 0]
            return [
                min(datos),
                max(datos),
                statistics.mean(datos),
                statistics.stdev(datos) if len(datos) > 1 else 0
            ]

        stats_fitness = calcular_estadisticas(fitness_results)
        stats_time = calcular_estadisticas(time_results)

        # Extracción de nombre y tamaño
        if '_' in algoritmo['name']:
            nombre_base, tamaño = algoritmo['name'].split('_', 1)
        else:
            nombre_base = algoritmo['name']
            tamaño = 'Desconocido'

        # Escritura de resultados
        with open(path_calidad, 'a', newline='') as f:
            csv.writer(f).writerow([nombre_base, tamaño] + stats_fitness)
        
        with open(path_tiempo, 'a', newline='') as f:
            csv.writer(f).writerow([nombre_base, tamaño] + stats_time)
        
        # Cálculo de estabilidad (coeficiente de variación)
        media = stats_fitness[2]
        desviacion = stats_fitness[3]
        cv = (desviacion/media)*100 if media != 0 else 0  # En porcentaje
        
        with open(path_estabilidad, 'a', newline='') as f:
            csv.writer(f).writerow([nombre_base, tamaño, round(cv, 2)])

if __name__ == "__main__":
    realizar_comparaciones()
    print("\nComparaciones completadas. Resultados guardados en archivos CSV.")