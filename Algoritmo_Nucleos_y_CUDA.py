import csv
import random
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd

# Wrapper para ejecutar el algoritmo con los parámetros dados
def ejecutar_algoritmo_wrapper(params, distancias, num_iteraciones):
    num_pob, num_gen, pm, m, num_competidores, hijos_crossover = params
    return ejecutar_algoritmo(num_pob, num_gen, pm, m, num_competidores, hijos_crossover, distancias, num_iteraciones)

# Función para ejecutar el algoritmo con parámetros específicos
def ejecutar_con_parametros(params, distancias, num_iteraciones):
    return ejecutar_algoritmo_wrapper(params, distancias, num_iteraciones)

# Producto cartesiano sin itertools
def cartesian_product(lists):
    if len(lists) == 0:
        return [[]]
    return [[x] + rest for x in lists[0] for rest in cartesian_product(lists[1:])]

# Función principal del algoritmo genético
def ejecutar_algoritmo(num_pob, num_gen, pm, m, num_competidores, hijos_crossover, distancias, num_iteraciones):
    num_var = len(distancias)
    resultados_generales = []

    for _ in range(num_iteraciones):
        poblacion = [random.sample(range(num_var), num_var) for _ in range(num_pob)]
        aptitudes = [costo(ind, distancias) for ind in poblacion]
        mejor_aptitud_historico = min(aptitudes)
        mejor_individuo_historico = poblacion[aptitudes.index(mejor_aptitud_historico)]

        for _ in range(num_gen):
            padres = seleccion_torneo(poblacion, aptitudes, num_competidores)
            hijos = []
            aptitudes_hijos = []

            for i in range(0, num_pob, 2):
                if hijos_crossover == 1:
                    hijo = cycle_crossover(padres[i], padres[i + 1])
                    hijo = heuristica_abruptos(hijo, m, distancias)
                    aptitudes_hijos.append(costo(hijo, distancias))
                    hijos.append(hijo)
                else:
                    hijo1 = cycle_crossover(padres[i], padres[i + 1])
                    hijo2 = cycle_crossover(padres[i + 1], padres[i])
                    hijo1 = heuristica_abruptos(hijo1, m, distancias)
                    hijo2 = heuristica_abruptos(hijo2, m, distancias)
                    hijos.extend([hijo1, hijo2])
                    aptitudes_hijos.extend([costo(hijo1, distancias), costo(hijo2, distancias)])

            poblacion = hijos
            aptitudes = aptitudes_hijos

            for i in range(num_pob):
                if random.random() < pm:
                    poblacion[i] = mutacion(poblacion[i])
                    aptitudes[i] = costo(poblacion[i], distancias)

            mejor_aptitud_generacion = min(aptitudes)
            if mejor_aptitud_generacion < mejor_aptitud_historico:
                mejor_aptitud_historico = mejor_aptitud_generacion
                mejor_individuo_historico = poblacion[aptitudes.index(mejor_aptitud_generacion)]

        resultados_generales.append(mejor_aptitud_historico)

    return min(resultados_generales), sum(resultados_generales) / len(resultados_generales), max(resultados_generales), (
        sum((x - sum(resultados_generales) / len(resultados_generales)) ** 2 for x in resultados_generales) /
        len(resultados_generales)) ** 0.5, time.time()

# Función de costos
def costo(individuo, distancias):
    print(individuo)
    print(len(individuo))
    total = 0
    for i in range(len(individuo)-2):
        total += distancias[individuo[i]][individuo[(i + 1)]]  
    total += distancias[individuo[31]][individuo[(0) % len(individuo)]]
    return total

# Selección por torneo
def seleccion_torneo(poblacion, aptitudes, num_competidores):
    padres = []
    for _ in range(len(poblacion)):
        competidores = random.sample(range(len(poblacion)), num_competidores)
        mejor_competidor = min(competidores, key=lambda x: aptitudes[x])
        padres.append(poblacion[mejor_competidor])
    return padres

# Crossover de ciclos
def cycle_crossover(padre1, padre2):
    hijo = [-1] * len(padre1)
    visitado = [False] * len(padre1)
    ciclo = 0

    while not all(visitado):
        inicio = visitado.index(False)
        ciclo += 1
        actual = inicio

        while True:
            hijo[actual] = padre1[actual] if ciclo % 2 == 1 else padre2[actual]
            visitado[actual] = True
            siguiente = padre1.index(padre2[actual])
            if visitado[siguiente] or siguiente == inicio:
                break
            actual = siguiente
    return hijo

# Mutación
def mutacion(individuo):
    idx1, idx2 = random.sample(range(len(individuo)), 2)
    individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
    return individuo

# Heurística para evitar abruptos
def heuristica_abruptos(hijo, m, distancias):
    for i in range(len(hijo)):
        distancias_ordenadas = sorted(range(len(distancias[i])), key=lambda x: distancias[i][x])
        vecinos = distancias_ordenadas[1:m + 1]
        mejor_costo = float('inf')
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

# Función que pasará parámetros a la función wrapper
def ejecutar_algoritmo_con_parametros(params):
    distancias = params[0]
    num_iteraciones = params[1]
    return ejecutar_algoritmo_wrapper(params[2], distancias, num_iteraciones)

# Función principal
if __name__ == "__main__":
    # Leer el archivo CSV y cargar los datos
    try:
        data = np.loadtxt('Distancias_no_head.csv', delimiter=',')
        distancias = data[:, 1:]  # Eliminar la primera columna si contiene etiquetas
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        exit(1)

    num_iteraciones = 10

    param_ranges = [
        [20, 50, 100],
        [50, 100, 200],
        [0.05, 0.1, 0.2],
        [2, 3, 5],
        [2, 3, 5],
        [1, 2]
    ]

    combinaciones = cartesian_product(param_ranges)
    resultados = []

    num_workers = 10  # Número de núcleos a usar
    with Pool(num_workers) as pool:
        # Pasamos los parámetros adicionales (distancias y num_iteraciones) a la función wrapper
        resultados = pool.map(ejecutar_algoritmo_con_parametros, [(distancias, num_iteraciones, params) for params in combinaciones])

    # Convertir los resultados en una estructura organizada
    resultados_array = np.array(resultados)
    nombres_columnas = ['Num_Pob', 'Num_Gen', 'Pm', 'm', 'Num_Competidores', 'Hijos_Crossover', 'Mejor', 'Media', 'Peor', 'Desviacion', 'Tiempo']

    # Convertir resultados a un DataFrame para análisis (opcional)
    df_resultados = pd.DataFrame(resultados_array, columns=nombres_columnas)

    # Análisis de resultados
    mejor_tiempo = df_resultados.nsmallest(5, 'Tiempo')
    mejor_valor = df_resultados.nsmallest(5, 'Mejor')
    menor_desviacion = df_resultados.nsmallest(5, 'Desviacion')

    # Métrica combinada
    df_resultados['Metrica_Compuesta'] = -df_resultados['Mejor'] / df_resultados['Mejor'].max() - \
                                         df_resultados['Tiempo'] / df_resultados['Tiempo'].max() - \
                                         df_resultados['Desviacion'] / df_resultados['Desviacion'].max()
    mejor_combinada = df_resultados.nlargest(5, 'Metrica_Compuesta')

    # Mostrar resultados
    print("Menor Tiempo:")
    print(mejor_tiempo)

    print("Menor Valor de 'Mejor':")
    print(mejor_valor)

    print("Menor Desviación:")
    print(menor_desviacion)

    print("Métrica Combinada:")
    print(mejor_combinada)
