import random
from multiprocessing import Pool
# Quitar las librerias de abajo en el algortmo puro
import time
import statistics
import pandas as pd

# Ejecuta una combinación de parámetros y devuelve los resultados.
def ejecutar_combinacion(param):
    distancias, combinacion, idx, total = param
    num_pob, num_gen, pm, m, num_competidores, hijos_crossover = combinacion

    iter_params = [
        (num_pob, num_gen, pm, m, num_competidores, hijos_crossover, distancias)
        for _ in range(10)
    ]

    # Medir el tiempo de ejecución
    start_time = time.time()

    # Ejecutar iteraciones en paralelo
    with Pool(10) as pool:
        resultados = pool.map(ejecutar_iteracion, iter_params)

    # Calcular métricas
    mejor = min(resultados)
    promedio = sum(resultados) / len(resultados)
    peor = max(resultados)
    desviacion = statistics.stdev(resultados)
    tiempo_total = time.time() - start_time

    # Mostrar el progreso de las combinaciones ejecutadas
    print(f"{idx}/{total}")
    return [num_pob, num_gen, pm, m, num_competidores, hijos_crossover, 
            mejor, promedio, peor, desviacion, tiempo_total]

# Ejecuta una iteración individual del algoritmo genético
def ejecutar_iteracion(params):
    return ejecutar_algoritmo(*params)

# Ejecución principal del algoritmo genético
def ejecutar_algoritmo(num_pob, num_gen, pm, m, num_competidores, hijos_crossover, distancias):
    num_var = len(distancias)
    poblacion = [random.sample(range(num_var), num_var) for _ in range(num_pob)]
    aptitudes = [costo(ind, distancias) for ind in poblacion]

    mejor_aptitud_historico, idx = min((apt, idx) for idx, apt in enumerate(aptitudes))
    mejor_individuo_historico = poblacion[idx]

    for _ in range(num_gen):
        
        # Inicializar y seleccionar los padres para el cruzamiento
        padres = seleccion_torneo(poblacion, aptitudes, num_competidores)
        
        # Inicializar las variables de los hijos y sus aptitudes
        hijos = []
        aptitudes_hijos = []

        # Realizar cruzamiento y reemplazar la población con los hijos
        poblacion, aptitudes = cruzamiento(num_pob,padres,hijos_crossover,m,hijos,aptitudes_hijos)

        # Aplicar mutación en la población
        poblacion, aptitudes = mutacion(poblacion, aptitudes, pm, distancias)

        # Actualizar el mejor histórico
        mejor_aptitud_generacion, idx = min((apt, idx) for idx, apt in enumerate(aptitudes))
        if mejor_aptitud_generacion < mejor_aptitud_historico:
            mejor_aptitud_historico = mejor_aptitud_generacion
            mejor_individuo_historico = poblacion[idx]

    return mejor_aptitud_historico

# Funciones básicas del algoritmo
def cruzamiento(num_pob,padres,hijos_crossover,m,hijos,aptitudes_hijos):
    for i in range(0, num_pob, 2):
            if hijos_crossover == 1:
                # Generar un único hijo
                hijo1 = heuristica_abruptos(cycle_crossover(padres[i], padres[i + 1]), m, distancias)
                aptitud_hijo1 = costo(hijo1, distancias)

                # Comparar con los padres y elegir los dos mejores
                individuos = [padres[i], padres[i + 1], hijo1]
                aptitudes_individuos = [
                    costo(padres[i], distancias),
                    costo(padres[i + 1], distancias),
                    aptitud_hijo1,
                ]
                mejores_indices = sorted(
                    range(len(aptitudes_individuos)),
                    key=lambda x: aptitudes_individuos[x],
                )[:2]
                mejores_individuos = [individuos[idx] for idx in mejores_indices]

                hijos.extend(mejores_individuos)
                aptitudes_hijos.extend([aptitudes_individuos[idx] for idx in mejores_indices])

            else:
                # Generar dos hijos
                hijo1 = heuristica_abruptos(cycle_crossover(padres[i], padres[i + 1]), m, distancias)
                hijo2 = heuristica_abruptos(cycle_crossover(padres[i + 1], padres[i]), m, distancias)
                aptitud_hijo1 = costo(hijo1, distancias)
                aptitud_hijo2 = costo(hijo2, distancias)

                # Comparar con los padres y elegir los dos mejores
                individuos = [padres[i], padres[i + 1], hijo1, hijo2]
                aptitudes_individuos = [
                    costo(padres[i], distancias),
                    costo(padres[i + 1], distancias),
                    aptitud_hijo1,
                    aptitud_hijo2,
                ]
                mejores_indices = sorted(
                    range(len(aptitudes_individuos)),
                    key=lambda x: aptitudes_individuos[x],
                )[:2]
                mejores_individuos = [individuos[idx] for idx in mejores_indices]

                hijos.extend(mejores_individuos)
                aptitudes_hijos.extend([aptitudes_individuos[idx] for idx in mejores_indices])
    return hijos, aptitudes_hijos

def mutacion(poblacion, aptitudes, pm, distancias):
    for j in range(len(poblacion)):
            if random.random() < pm:
                idx1, idx2 = random.sample(range(len(poblacion[j])), 2)
                poblacion[j][idx1], poblacion[j][idx2] = poblacion[j][idx2], poblacion[j][idx1]
                aptitudes[j] = costo(poblacion[j], distancias)

def seleccion_torneo(poblacion, aptitudes, num_competidores):
    return [
        poblacion[
            min(random.sample(range(len(poblacion)), num_competidores), key=aptitudes.__getitem__)
        ]
        for _ in range(len(poblacion))
    ]


# Funciones auxiliares
def costo(individuo, distancias):
    return sum(
        distancias[individuo[i]][individuo[i + 1]] for i in range(len(individuo) - 1)
    ) + distancias[individuo[-1]][individuo[0]]

def cycle_crossover(padre1, padre2):
    hijo = [-1] * len(padre1)
    ciclo = 0
    visitado = [False] * len(padre1)

    while not all(visitado):
        inicio = visitado.index(False)
        ciclo += 1
        actual = inicio

        while True:
            hijo[actual] = padre1[actual] if ciclo % 2 else padre2[actual]
            visitado[actual] = True
            siguiente = padre1.index(padre2[actual])
            if visitado[siguiente]:
                break
            actual = siguiente
    return hijo

def heuristica_abruptos(hijo, m, distancias):
    for i in range(len(hijo)):
        distancias_ordenadas = sorted(range(len(distancias[i])), key=lambda x: distancias[i][x])
        vecinos = distancias_ordenadas[1 : m + 1]
        mejor_costo = float("inf")
        mejor_ruta = hijo[:]
        for vecino in vecinos:
            ruta = hijo[:]
            ruta.remove(i)
            ruta.insert(ruta.index(vecino), i)
            costo_ruta = costo(ruta, distancias)
            if costo_ruta < mejor_costo:
                mejor_costo = costo_ruta
                mejor_ruta = ruta
        hijo = mejor_ruta
    return hijo

if __name__ == "__main__":
    # Configuración de entrada
    try:
        # Abrir el archivo en modo de lectura
        with open('Distancias_no_head.csv', 'r') as archivo:
            # Leer todas las líneas del archivo
            lineas = archivo.readlines()

        # Crear una matriz vacía para almacenar las distancias
        distancias = []

        # Recorrer cada línea del archivo
        for linea in lineas:
            # Dividir la línea en elementos usando la coma como delimitador
            fila = [float(valor.strip()) for valor in linea.split(',')]
            # Agregar la fila a la matriz de distancias
            distancias.append(fila)

    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        exit(1)

    # Establecer los valores de los parametros a testear
    # Num de Población
    # Num de Generaciones
    # Probabilidad de mutación
    # Parametro de m para la heuristica
    # Numero de competidores para el torneo de selección de padres
    # Número de hijos a producir en el Cycle Crossover
    param_ranges = [
        [20, 50, 100],
        [50, 100, 200],
        [0.05, 0.1, 0.2],
        [2, 3, 5],
        [2, 3, 5],
        [1, 2],
    ]

    combinaciones = [
        [x, y, z, a, b, c]
        for x in param_ranges[0]
        for y in param_ranges[1]
        for z in param_ranges[2]
        for a in param_ranges[3]
        for b in param_ranges[4]
        for c in param_ranges[5]
    ]

    num_combinaciones = len(combinaciones)
    params = [
        (distancias, combinacion, idx + 1, num_combinaciones)
        for idx, combinacion in enumerate(combinaciones)
    ]

    resultados_generales = []

    for param in params:
        resultados_generales.append(ejecutar_combinacion(param))

    columns = ["Num_Pob", "Num_Gen", "Pm", "m", "Num_Competidores", "Hijos_Crossover", 
            "Mejor", "Media", "Peor", "Desviacion", "Tiempo"]

    resultados_df = pd.DataFrame(resultados_generales, columns=columns)

    # Guardar en un archivo Excel
    output_file = "resultados_generales_CPU.xlsx"
    resultados_df.to_excel(output_file, index=False)

    print(f"Resultados guardados en el archivo: {output_file}")