# recocido.py
import ctypes
from ctypes import c_int, c_double, c_char_p, POINTER, Structure, c_char
import os
import matplotlib.pyplot as plt

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
            c_int,    # longitud_ruta
            c_int,    # num_generaciones
            c_double, # tasa_enfriamiento
            c_double, # temperatura_final
            c_int,    # max_neighbours
            c_int,    # m
            c_char_p, # nombre_archivo
        ]
        self.lib.liberar_resultado_recocido.argtypes = [POINTER(ResultadoRecocido)]

    def ejecutar(self, longitud_ruta, num_generaciones,
                 tasa_enfriamiento, temperatura_final,
                 max_neighbours, m, nombre_archivo):
        ptr = self.lib.ejecutar_algoritmo_recocido(
            longitud_ruta,
            num_generaciones,
            tasa_enfriamiento,
            temperatura_final,
            max_neighbours,
            m,
            nombre_archivo.encode('utf-8')
        )
        res = ptr.contents

        gens = [res.fitness_generaciones[i] for i in range(num_generaciones)]
        nombres = [ res.nombres_ciudades[i].value.decode('utf-8')
                    for i in range(res.longitud_recorrido) ]
        ruta   = [res.recorrido[i] for i in range(res.longitud_recorrido)]

        self.lib.liberar_resultado_recocido(ptr)
        return {
            "recorrido": ruta,
            "nombres_ciudades": nombres,
            "fitness": res.fitness,
            "tiempo_ejecucion": res.tiempo_ejecucion,
            "fitness_generaciones": gens
        }

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    lib  = "recocido.dll" if os.name=='nt' else "librecocido.so"
    ruta = os.path.join(base, lib)

    sa = AlgoritmoRecocido(ruta)
    out = sa.ejecutar(
        longitud_ruta=32,
        num_generaciones=75,
        tasa_enfriamiento=0.92,
        temperatura_final=0.001,
        max_neighbours=1000,
        m=3,
        nombre_archivo="Distancias_no_head.csv"
    )

    plt.plot(out["fitness_generaciones"])
    plt.title("Evolución del Fitness en Recocido Simulado")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.show()

    print(f"Mejor fitness: {out['fitness']:.2f}")
    print(f"Tiempo: {out['tiempo_ejecucion']:.2f}s")
    print("Ruta:", " -> ".join(out["nombres_ciudades"])+" -> "+out["nombres_ciudades"][0])

if __name__ == "__main__":
    main()
