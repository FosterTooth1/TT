import numpy as np
import time
import pandas as pd
from numba import njit

@njit
def costo(individuo, distancias):
    total_cost = 0.0
    for i in range(len(individuo) - 1):
        total_cost += distancias[individuo[i], individuo[i + 1]]
    total_cost += distancias[individuo[-1], individuo[0]]
    return total_cost

@njit
def cycle_crossover(padre1, padre2):
    n = len(padre1)
    hijo = np.full(n, -1, dtype=np.int64)
    visitado = np.zeros(n, dtype=np.bool_)

    ciclo = 0
    for inicio in range(n):
        if not visitado[inicio]:
            ciclo += 1
            actual = inicio
            while not visitado[actual]:
                visitado[actual] = True
                hijo[actual] = padre1[actual] if ciclo % 2 else padre2[actual]
                actual = np.where(padre1 == padre2[actual])[0][0]
                if visitado[actual]:
                    break
    return hijo

@njit
def heuristica_abruptos(hijo, m, distancias):
    mejor_ruta = hijo.copy()

    for i in range(len(hijo)):
        distancias_ciudad = distancias[i, :]
        indices_ordenados = np.argsort(distancias_ciudad)[1:m+1]
        mejor_costo_actual = costo(mejor_ruta, distancias)

        for vecino in indices_ordenados:
            ruta_temp = mejor_ruta.copy()
            idx_ciudad = np.where(ruta_temp == i)[0][0]
            idx_vecino = np.where(ruta_temp == vecino)[0][0]

            ruta_temp[idx_ciudad], ruta_temp[idx_vecino] = ruta_temp[idx_vecino], ruta_temp[idx_ciudad]
            costo_ruta = costo(ruta_temp, distancias)

            if costo_ruta < mejor_costo_actual:
                mejor_ruta = ruta_temp
                mejor_costo_actual = costo_ruta
    return mejor_ruta

@njit
def cruzamiento(num_pob, padres, hijos_crossover, m, distancias):
    hijos = np.empty_like(padres)
    aptitudes_hijos = np.zeros(num_pob, dtype=np.float64)
    n = padres.shape[1]  # número de ciudades

    for i in range(0, num_pob, 2):
        if hijos_crossover == 1:
            hijo1 = heuristica_abruptos(
                cycle_crossover(padres[i], padres[i+1]),
                m,
                distancias
            )
            # Creamos un array con [padre1, padre2, hijo1]
            individuos = np.empty((3, n), dtype=np.int64)
            individuos[0, :] = padres[i]
            individuos[1, :] = padres[i+1]
            individuos[2, :] = hijo1

            aptitudes_individuos = np.array([
                costo(padres[i], distancias),
                costo(padres[i+1], distancias),
                costo(hijo1, distancias)
            ], dtype=np.float64)

            indices_mejores = np.argsort(aptitudes_individuos)[:2]
            hijos[i] = individuos[indices_mejores[0]]
            hijos[i+1] = individuos[indices_mejores[1]]
            aptitudes_hijos[i] = aptitudes_individuos[indices_mejores[0]]
            aptitudes_hijos[i+1] = aptitudes_individuos[indices_mejores[1]]
        else:
            hijo1 = heuristica_abruptos(
                cycle_crossover(padres[i], padres[i+1]),
                m,
                distancias
            )
            hijo2 = heuristica_abruptos(
                cycle_crossover(padres[i+1], padres[i]),
                m,
                distancias
            )

            # Creamos un array con [padre1, padre2, hijo1, hijo2]
            individuos = np.empty((4, n), dtype=np.int64)
            individuos[0, :] = padres[i]
            individuos[1, :] = padres[i+1]
            individuos[2, :] = hijo1
            individuos[3, :] = hijo2

            aptitudes_individuos = np.array([
                costo(padres[i], distancias),
                costo(padres[i+1], distancias),
                costo(hijo1, distancias),
                costo(hijo2, distancias)
            ], dtype=np.float64)

            indices_mejores = np.argsort(aptitudes_individuos)[:2]
            hijos[i] = individuos[indices_mejores[0]]
            hijos[i+1] = individuos[indices_mejores[1]]
            aptitudes_hijos[i] = aptitudes_individuos[indices_mejores[0]]
            aptitudes_hijos[i+1] = aptitudes_individuos[indices_mejores[1]]

    return hijos, aptitudes_hijos

# Funciones con aleatoriedad SIN @njit
def random_permutation(n, rng):
    arr = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = rng.integers(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def seleccion_torneo(poblacion, aptitudes, num_competidores, rng):
    n = len(poblacion)
    padres = np.empty_like(poblacion)
    for i in range(n):
        indices_torneo = rng.choice(n, num_competidores, replace=False)
        aptitudes_torneo = aptitudes[indices_torneo]
        ganador = indices_torneo[np.argmin(aptitudes_torneo)]
        padres[i] = poblacion[ganador]
    return padres

def mutacion(poblacion, aptitudes, pm, distancias, rng):
    for j in range(len(poblacion)):
        if rng.random() < pm:
            idx1, idx2 = rng.choice(len(poblacion[j]), 2, replace=False)
            poblacion[j][idx1], poblacion[j][idx2] = poblacion[j][idx2], poblacion[j][idx1]
            aptitudes[j] = costo(poblacion[j], distancias)
    return poblacion, aptitudes

def ejecutar_algoritmo(num_pob, num_gen, pm, m, num_competidores, hijos_crossover, distancias):
    rng = np.random.default_rng(42)
    num_var = distancias.shape[0]

    # Población inicial
    poblacion = np.array([random_permutation(num_var, rng) for _ in range(num_pob)])
    aptitudes = np.array([costo(ind, distancias) for ind in poblacion])

    mejor_aptitud_historico = np.min(aptitudes)

    for _ in range(num_gen):
        # Selección
        padres = seleccion_torneo(poblacion, aptitudes, num_competidores, rng)

        # Cruzamiento
        poblacion, aptitudes = cruzamiento(num_pob, padres, hijos_crossover, m, distancias)

        # Mutación
        poblacion, aptitudes = mutacion(poblacion, aptitudes, pm, distancias, rng)

        # Actualizar mejor historial
        mejor_aptitud_generacion = np.min(aptitudes)
        if mejor_aptitud_generacion < mejor_aptitud_historico:
            mejor_aptitud_historico = mejor_aptitud_generacion

    return mejor_aptitud_historico

def main():
    
    tiempo_algoritmo = time.time()
    
    # Carga de datos
    try:
        distancias = np.loadtxt('Distancias_no_head.csv', delimiter=',')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    param_ranges = [
        [100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500],     # Population size
        [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500],    # Generations
        [0.05, 0.1, 0.15, 0.2, 0.25],  # Mutation probability
        [2, 3, 4, 5],         # Heuristic m
        [2, 3, 4, 5],         # Tournament competitors
        [1, 2]             # Crossover           # Crossover offspring
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

    resultados_generales = []

    for idx, combinacion in enumerate(combinaciones):
        start_time = time.time()
        iteraciones = [ejecutar_algoritmo(*combinacion, distancias) for _ in range(10)]

        mejor = min(iteraciones)
        promedio = np.mean(iteraciones)
        peor = max(iteraciones)
        desviacion = np.std(iteraciones)
        tiempo_total = time.time() - start_time

        print(f"{idx+1}/{len(combinaciones)}")

        resultados_generales.append(
            combinacion + [mejor, promedio, peor, desviacion, tiempo_total]
        )

    columns = ["Num_Pob", "Num_Gen", "Pm", "m", "Num_Competidores", "Hijos_Crossover",
               "Mejor", "Media", "Peor", "Desviacion", "Tiempo"]
    resultados_df = pd.DataFrame(resultados_generales, columns=columns)

    output_file = "resultados_generales_numba_test.xlsx"
    resultados_df.to_excel(output_file, index=False)
    print(f"Results saved to: {output_file}")

    tiempo_algoritmo_total = time.time() - tiempo_algoritmo
    print(tiempo_algoritmo_total)
    
if __name__ == "__main__":
    main()
