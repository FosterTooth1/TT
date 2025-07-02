import ctypes
from ctypes import c_int, c_double, c_char_p, c_char, POINTER, Structure, cast
import os
import matplotlib.pyplot as plt

# Definimos la estructura equivalente a ResultadoGenetico
class ResultadoPSO(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),          # Puntero al mejor recorrido
        ("fitness", c_double),                 # Valor de fitness
        ("tiempo_ejecucion", c_double),        # Tiempo de ejecución
        ("nombres_ciudades", POINTER(c_char * 50 * 32)),  # Array de nombres (32 ciudades)
        ("longitud_recorrido", c_int),          # Longitud del recorrido
        ("fitness_generaciones", POINTER(c_double))  # Evolución del fitness
    ]

# Clase wrapper para el algoritmo PSO
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

# Función main idéntica a la genética (solo cambian parámetros)
def main():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    nombre_biblioteca = "libpso.so" if os.name != 'nt' else "pso.dll"
    ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
    
    pso = AlgoritmoPSO(ruta_biblioteca)
    
    params = {
        'tamano_poblacion': 500,
        'longitud_ruta': 32,
        'num_generaciones': 150,
        'prob_pbest': 0.35,
        'prob_gbest': 0.7,
        'nombre_archivo': "Distancias_no_head.csv",
        'prob_inercia': 0.3,
        'm': 3,
        'heuristica': 0
    }
    
    resultado = pso.ejecutar(**params)
    
    print("\nMejor ruta PSO:")
    for i, (idx, nombre) in enumerate(zip(resultado['recorrido'], resultado['nombres_ciudades'])):
        print(f"{i+1}. {nombre} (índice: {idx})")
    print(f"\nFitness: {resultado['fitness']}")
    print(f"Tiempo: {resultado['tiempo_ejecucion']:.2f}s")
    
    plt.plot(resultado['fitness_generaciones'])
    plt.title("Evolución del Fitness en PSO")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()