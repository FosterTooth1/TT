import ctypes
from ctypes import c_int, c_double, c_float, c_char_p, POINTER, Structure, c_char, cast
import os
import matplotlib.pyplot as plt

class ResultadoTabu(Structure):
    _fields_ = [
        ("recorrido", POINTER(c_int)),
        ("fitness", c_double),
        ("tiempo_ejecucion", c_double),
        ("nombres_ciudades", POINTER(c_char * 50 * 32)),  # Mismo formato que otros
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
    ]

class AlgoritmoTabu:
    def __init__(self, ruta_biblioteca):
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        
        # Configuración estándar como en otros algoritmos
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
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoTabu)]  # Nombre unificado

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
            
            # Extracción de datos unificada
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

def main():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    nombre_biblioteca = "tabu.dll" if os.name == 'nt' else "libtabu.so"
    ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
    
    tabu = AlgoritmoTabu(ruta_biblioteca)
    
    params = {
        'longitud_ruta': 32,
        'tenencia_tabu': 7,
        'num_generaciones': 100,
        'max_neighbours': 500,
        'umbral_est_global': 0.1,
        'umbral_est_local': 0.05,
        'm': 3,
        'nombre_archivo': "Distancias_no_head.csv",
        'heuristica': 0
    }
    
    resultado = tabu.ejecutar(**params)
    
    print("\nMejor ruta Tabú:")
    for i, (idx, nombre) in enumerate(zip(resultado['recorrido'], resultado['nombres_ciudades'])):
        print(f"{i+1}. {nombre} (índice: {idx})")
    print(f"\nFitness: {resultado['fitness']:.2f}")
    print(f"Tiempo: {resultado['tiempo_ejecucion']:.2f}s")
    
    plt.plot(resultado['fitness_generaciones'])
    plt.title("Evolución del Fitness - Búsqueda Tabú")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()