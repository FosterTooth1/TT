import ctypes
from ctypes import c_int, c_double, c_float, c_char_p, POINTER, Structure, c_char
import os
import matplotlib.pyplot as plt

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
        # Definir firma de la función C
        self.lib.ejecutar_algoritmo_tabu.restype = POINTER(ResultadoTabu)
        self.lib.ejecutar_algoritmo_tabu.argtypes = [
            c_int,    # longitud_ruta
            c_int,    # tenencia_tabu
            c_int,    # num_generaciones
            c_int,    # max_neighbours
            c_float,  # umbral_est_global
            c_float,  # umbral_est_local
            c_int,    # m
            c_char_p, # nombre_archivo
            c_int     # heuristica
        ]
        self.lib.liberar_resultado_tabu.argtypes = [POINTER(ResultadoTabu)]

    def ejecutar(self, longitud_ruta, tenencia_tabu, num_generaciones,
                 max_neighbours, umbral_est_global, umbral_est_local,
                 m, nombre_archivo, heuristica):
        # Llamada al C
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

        # Extraer vectores
        recorrido = [res.recorrido[i] for i in range(res.longitud_recorrido)]
        nombres = [res.nombres_ciudades[i].value.decode('utf-8')
                   for i in range(res.longitud_recorrido)]
        fitness_gens = [res.fitness_generaciones[i]
                        for i in range(num_generaciones)]

        # Liberar memoria en C
        self.lib.liberar_resultado_tabu(ptr)

        return {
            "recorrido": recorrido,
            "nombres_ciudades": nombres,
            "fitness": res.fitness,
            "tiempo_ejecucion": res.tiempo_ejecucion,
            "fitness_generaciones": fitness_gens
        }


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    lib_name = "tabu.dll" if os.name == 'nt' else "libtabu.so"
    ruta = os.path.join(base, lib_name)

    tabu = AlgoritmoTabu(ruta)
    out = tabu.ejecutar(
        longitud_ruta=32,
        tenencia_tabu=7,
        num_generaciones=100,
        max_neighbours=500,
        umbral_est_global=0.1,
        umbral_est_local=0.05,
        m=3,
        nombre_archivo="Distancias_no_head.csv",
        heuristica=0
    )

    # Graficar evolución del fitness
    plt.plot(out["fitness_generaciones"])
    plt.title("Evolución del Fitness en Tabu Search")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.show()

    # Resultados
    print(f"Mejor fitness: {out['fitness']:.2f}")
    print(f"Tiempo de ejecución: {out['tiempo_ejecucion']:.2f}s")
    print("Ruta: ", " -> ".join(out['nombres_ciudades']) + " -> " + out['nombres_ciudades'][0])

if __name__ == "__main__":
    main()
