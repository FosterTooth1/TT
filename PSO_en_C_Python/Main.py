import ctypes
from ctypes import c_int, c_double, c_char_p, POINTER, Structure, c_char
import os
import matplotlib.pyplot as plt

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
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        self.biblioteca.ejecutar_algoritmo_pso.restype = POINTER(ResultadoPSO)
        self.biblioteca.ejecutar_algoritmo_pso.argtypes = [
            c_int, c_int, c_int, c_double, c_double, c_char_p, c_double, c_double
        ]
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoPSO)]
    
    def ejecutar(self, params):
        resultado_ptr = self.biblioteca.ejecutar_algoritmo_pso(
            params['tamano_poblacion'],
            params['longitud_ruta'],
            params['num_generaciones'],
            params['prob_pbest'],
            params['prob_gbest'],
            params['nombre_archivo'].encode('utf-8'),
            params['prob_heuristica'],
            params['prb_aleatorio'],
        )
        
        resultado = resultado_ptr.contents
        num_gen = params['num_generaciones']
        fitness_generaciones = [resultado.fitness_generaciones[i] for i in range(num_gen)]

        
        nombres_ciudades = []
        for i in range(resultado.longitud_recorrido):
            raw = resultado.nombres_ciudades[i]   # esto es un (c_char * 50)
            # .value te devuelve bytes hasta el primer '\0'
            nombre = raw.value.decode('utf-8')     
            nombres_ciudades.append(nombre)


        
        recorrido = [resultado.recorrido[i] for i in range(resultado.longitud_recorrido)]
        
        self.biblioteca.liberar_resultado(resultado_ptr)
        
        return {
            'recorrido': recorrido,
            'nombres_ciudades': nombres_ciudades,
            'fitness': resultado.fitness,
            'tiempo_ejecucion': resultado.tiempo_ejecucion,
            'fitness_generaciones': fitness_generaciones
        }

def main():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    nombre_biblioteca = "libpso.so" if os.name != 'nt' else "pso.dll"
    ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
    
    pso = AlgoritmoPSO(ruta_biblioteca)
    
    params = {
        'tamano_poblacion': 50,
        'longitud_ruta': 32,
        'num_generaciones': 300,
        'prob_pbest': 0.5,
        'prob_gbest': 0.6,
        'nombre_archivo': "Distancias_no_head.csv",
        'prob_heuristica': 0.7, 
        'prb_aleatorio': 0.5,
    }
    
    resultado = pso.ejecutar(params)
    
    # Graficar fitness por generación
    plt.plot(resultado['fitness_generaciones'])
    plt.title("Evolución del Fitness en PSO")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.show()
    
    print(f"Mejor Fitness: {resultado['fitness']}")
    print(f"Tiempo: {resultado['tiempo_ejecucion']} segundos")

if __name__ == "__main__":
    main()