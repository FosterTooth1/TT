import os
import matplotlib.pyplot as plt
import ctypes
from ctypes import (
    CDLL, Structure, POINTER,
    c_int, c_double, c_float,
    c_char_p, c_char
)

class ResultadoPSO(Structure):
    _fields_ = [
        ("recorrido",          POINTER(c_int)),
        ("fitness",            c_double),
        ("tiempo_ejecucion",   c_double),
        ("nombres_ciudades",   POINTER(c_char * 50)),
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
    ]

class AlgoritmoPSO:
    def __init__(self, ruta_biblioteca):
        self.bibl = CDLL(ruta_biblioteca)
        self.bibl.ejecutar_algoritmo_pso.restype = POINTER(ResultadoPSO)
        self.bibl.ejecutar_algoritmo_pso.argtypes = [
            c_int, c_int, c_int,
            c_double, c_double,
            c_char_p,
            c_double,
            c_int, c_int
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

        # fitness_generaciones
        fg = [res.fitness_generaciones[i] for i in range(params['num_generaciones'])]

        # nombres_ciudades
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

def main():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    nombre_biblioteca = "libpso.so" if os.name != 'nt' else "pso.dll"
    ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
    
    pso = AlgoritmoPSO(ruta_biblioteca)
    
    params = {
        'tamano_poblacion': 20000,
        'longitud_ruta': 32,
        'num_generaciones': 500,
        'prob_pbest': 0.35,
        'prob_gbest': 0.7,
        'nombre_archivo': "Distancias_no_head.csv",
        'prob_inercia': 0.3,
        'm': 3,
        'heuristica': 0,
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