import random
import cupy as cp
import time
import statistics
import pandas as pd

def ejecutar_combinacion(param):
    """Ejecuta una combinación de parámetros y devuelve los resultados."""
    distancias, combinacion, idx, total = param
    num_pob, num_gen, pm, m, num_competidores, hijos_crossover = combinacion

    iter_params = [
        (num_pob, num_gen, pm, m, num_competidores, hijos_crossover, distancias)
        for _ in range(10)
    ]

    print("Iniciando las ejecuciones de combinaciones...")
    
    # Medir el tiempo de ejecución
    start_time = time.time()

    # Ejecutar iteraciones en paralelo (en GPU con CuPy)
    resultados = [ejecutar_iteracion(params) for params in iter_params]

    # Calcular métricas
    mejor = min(resultados)
    promedio = sum(resultados) / len(resultados)
    peor = max(resultados)
    desviacion = statistics.stdev(resultados)
    tiempo_total = time.time() - start_time

    print(f"{idx}/{total}")
    return [num_pob, num_gen, pm, m, num_competidores, hijos_crossover, 
            mejor, promedio, peor, desviacion, tiempo_total]

def ejecutar_iteracion(params):
    """Ejecuta una iteración individual del algoritmo genético."""
    return ejecutar_algoritmo(*params)

def ejecutar_algoritmo(num_pob, num_gen, pm, m, num_competidores, hijos_crossover, distancias_gpu):
    """Ejecución principal del algoritmo genético."""
    num_var = len(distancias)
    poblacion = [random.sample(range(num_var), num_var) for _ in range(num_pob)]
    
    # Evaluar aptitudes en GPU
    aptitudes = costo_gpu(poblacion, distancias_gpu)

    mejor_aptitud_historico, idx = min((apt, idx) for idx, apt in enumerate(aptitudes))
    mejor_individuo_historico = poblacion[idx]

    for _ in range(num_gen):
        padres = seleccion_torneo_gpu(poblacion, aptitudes, num_competidores)
        hijos = []
        aptitudes_hijos = []

        for i in range(0, num_pob, 2):
            if hijos_crossover == 1:
                hijo1 = heuristica_abruptos(cycle_crossover(padres[i], padres[i + 1]), m, distancias_gpu)
                aptitud_hijo1 = costo_individuo_gpu(hijo1, distancias_gpu)

                individuos = [padres[i], padres[i + 1], hijo1]
                aptitudes_individuos = [
                    costo_individuo_gpu(padres[i], distancias_gpu),
                    costo_individuo_gpu(padres[i + 1], distancias_gpu),
                    aptitud_hijo1,
                ]
                mejores_indices = sorted(range(len(aptitudes_individuos)), key=lambda x: aptitudes_individuos[x])[:2]
                mejores_individuos = [individuos[idx] for idx in mejores_indices]

                hijos.extend(mejores_individuos)
                aptitudes_hijos.extend([aptitudes_individuos[idx] for idx in mejores_indices])

            else:
                hijo1 = heuristica_abruptos(cycle_crossover(padres[i], padres[i + 1]), m, distancias_gpu)
                hijo2 = heuristica_abruptos(cycle_crossover(padres[i + 1], padres[i]), m, distancias_gpu)
                aptitud_hijo1 = costo_individuo_gpu(hijo1, distancias_gpu)
                aptitud_hijo2 = costo_individuo_gpu(hijo2, distancias_gpu)

                individuos = [padres[i], padres[i + 1], hijo1, hijo2]
                aptitudes_individuos = [
                    costo_individuo_gpu(padres[i], distancias_gpu),
                    costo_individuo_gpu(padres[i + 1], distancias_gpu),
                    aptitud_hijo1,
                    aptitud_hijo2,
                ]
                mejores_indices = sorted(range(len(aptitudes_individuos)), key=lambda x: aptitudes_individuos[x])[:2]
                mejores_individuos = [individuos[idx] for idx in mejores_indices]

                hijos.extend(mejores_individuos)
                aptitudes_hijos.extend([aptitudes_individuos[idx] for idx in mejores_indices])

        poblacion = hijos
        aptitudes = aptitudes_hijos

        for j in range(len(poblacion)):
            if random.random() < pm:
                poblacion[j] = mutacion(poblacion[j])
                aptitudes[j] = costo_individuo_gpu(poblacion[j], distancias_gpu)

        mejor_aptitud_generacion, idx = min((apt, idx) for idx, apt in enumerate(aptitudes))
        if mejor_aptitud_generacion < mejor_aptitud_historico:
            mejor_aptitud_historico = mejor_aptitud_generacion
            mejor_individuo_historico = poblacion[idx]

    return float(mejor_aptitud_historico)

def costo_gpu(poblacion, distancias_gpu):
    """Calcula los costos en paralelo usando CuPy, optimizado."""
    poblacion_gpu = cp.array(poblacion)
    indices_1 = poblacion_gpu[:, :-1]
    indices_2 = poblacion_gpu[:, 1:]
    
    # Calcular costos en GPU vectorizado
    costos = cp.sum(distancias_gpu[indices_1, indices_2], axis=1)
    costos += distancias_gpu[poblacion_gpu[:, -1], poblacion_gpu[:, 0]]
    
    return costos


def costo_individuo_gpu(individuo, distancias_gpu):
    """Calcula el costo de un individuo específico en GPU."""
    individuo_gpu = cp.array(individuo)
    return cp.sum(
        distancias_gpu[individuo_gpu[:-1], individuo_gpu[1:]]
    ) + distancias_gpu[individuo_gpu[-1], individuo_gpu[0]]

def seleccion_torneo_gpu(poblacion, aptitudes, num_competidores):
    return [
        poblacion[
            min(random.sample(range(len(poblacion)), num_competidores), key=lambda idx: aptitudes[idx])
        ]
        for _ in range(len(poblacion))
    ]

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

def mutacion(individuo):
    idx1, idx2 = random.sample(range(len(individuo)), 2)
    individuo[idx1], individuo[idx2] = individuo[idx2], individuo[idx1]
    return individuo

def heuristica_abruptos(hijo, m, distancias_gpu):
    for i in range(len(hijo)):
        distancias_ordenadas = cp.argsort(distancias_gpu[i])[:m + 1].tolist()
        vecinos = distancias_ordenadas[1:]
        mejor_costo = float("inf")
        mejor_ruta = hijo[:]
        for vecino in vecinos:
            ruta = hijo[:]
            ruta.remove(i)
            ruta.insert(ruta.index(vecino), i)
            costo_ruta = costo_individuo_gpu(ruta, distancias_gpu)
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

    # Rango de parámetros para probar combinaciones
    param_ranges = [
        [20, 50, 100],  # Tamaños de población
        [50, 100, 200],  # Número de generaciones
        [0.05, 0.1, 0.2],  # Probabilidad de mutación
        [2, 3, 5],  # Número de vecinos para heurística
        [2, 3, 5],  # Número de competidores para selección de torneo
        [1, 2],  # Número de hijos generados en crossover
    ]

    # Generar todas las combinaciones posibles de parámetros
    combinaciones = [
        [x, y, z, a, b, c]
        for x in param_ranges[0]
        for y in param_ranges[1]
        for z in param_ranges[2]
        for a in param_ranges[3]
        for b in param_ranges[4]
        for c in param_ranges[5]
    ]

    # Contar el número total de combinaciones
    num_combinaciones = len(combinaciones)
    
    distancias= cp.array(distancias)
    
    # Preparar los parámetros para pasar a cada ejecución
    params = [
        (distancias, combinacion, idx + 1, num_combinaciones)
        for idx, combinacion in enumerate(combinaciones)
    ]

    # Almacenar los resultados de todas las ejecuciones
    resultados_generales = []

    # Ejecutar todas las combinaciones
    for param in params:
        resultados_generales.append(ejecutar_combinacion(param))

    # Definir las columnas del DataFrame
    columns = ["Num_Pob", "Num_Gen", "Pm", "m", "Num_Competidores", "Hijos_Crossover", 
               "Mejor", "Media", "Peor", "Desviacion", "Tiempo"]

    # Crear un DataFrame con los resultados
    resultados_df = pd.DataFrame(resultados_generales, columns=columns)

    # Guardar los resultados en un archivo Excel
    output_file = "resultados_generales_GPU.xlsx"
    resultados_df.to_excel(output_file, index=False)

    print(f"Resultados guardados en el archivo: {output_file}")
