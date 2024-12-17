import random
import time
import statistics
import pandas as pd
from numba import njit, prange, int64, float64

# Optimized fitness function
@njit
def costo(individuo, distancias):
    total = 0.0
    for i in range(len(individuo) - 1):
        total += distancias[individuo[i]][individuo[i + 1]]
    total += distancias[individuo[-1]][individuo[0]]
    return total

# Tournament selection with Numba optimization
@njit
def encontrar_mejor_competidor(poblacion, aptitudes, competidores):
    mejor_idx = competidores[0]
    for idx in competidores[1:]:
        if aptitudes[idx] < aptitudes[mejor_idx]:
            mejor_idx = idx
    return mejor_idx

@njit
def seleccion_torneo(poblacion, aptitudes, num_competidores):
    nuevos_padres = []
    for _ in range(len(poblacion)):
        # Select random competitors
        competidores = []
        while len(competidores) < num_competidores:
            nuevo_competidor = random.randint(0, len(poblacion) - 1)
            if nuevo_competidor not in competidores:
                competidores.append(nuevo_competidor)
        
        # Find the best competitor
        mejor_competidor = encontrar_mejor_competidor(poblacion, aptitudes, competidores)
        nuevos_padres.append(poblacion[mejor_competidor])
    
    return nuevos_padres

# Cycle crossover with Numba optimization
@njit
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
            
            # Find the index of the next element in the cycle
            siguiente = -1
            for j in range(len(padre1)):
                if padre1[j] == padre2[actual] if ciclo % 2 else padre2[j] == padre1[actual]:
                    siguiente = j
                    break
            
            if siguiente == -1 or visitado[siguiente]:
                break
            actual = siguiente
    
    return hijo

# Mutation with Numba optimization
@njit
def mutacion(poblacion, aptitudes, pm, distancias):
    for j in prange(len(poblacion)):
        if random.random() < pm:
            # Swap two random indices
            idx1 = random.randint(0, len(poblacion[j]) - 1)
            idx2 = random.randint(0, len(poblacion[j]) - 1)
            while idx2 == idx1:
                idx2 = random.randint(0, len(poblacion[j]) - 1)
            
            poblacion[j][idx1], poblacion[j][idx2] = poblacion[j][idx2], poblacion[j][idx1]
            
            # Recalculate fitness
            aptitudes[j] = costo(poblacion[j], distancias)
    
    return poblacion, aptitudes

# Heuristic local search with Numba optimization
@njit
def encontrar_mejor_ruta(hijo, m, distancias):
    mejor_costo = costo(hijo, distancias)
    mejor_ruta = hijo.copy()

    for i in range(len(hijo)):
        # Find distances from current city
        distancias_ciudad = distancias[i]
        
        # Create sorted indices of distances
        distancias_ordenadas = list(range(len(distancias_ciudad)))
        
        # Custom sorting without lambda
        for k in range(len(distancias_ordenadas)):
            for l in range(k + 1, len(distancias_ordenadas)):
                if (distancias_ciudad[distancias_ordenadas[k]] > distancias_ciudad[distancias_ordenadas[l]] 
                    or (distancias_ciudad[distancias_ordenadas[k]] == distancias_ciudad[distancias_ordenadas[l]] 
                        and distancias_ordenadas[k] > distancias_ordenadas[l])):
                    distancias_ordenadas[k], distancias_ordenadas[l] = distancias_ordenadas[l], distancias_ordenadas[k]
        
        # Get potential neighbors (excluding self)
        vecinos = [v for v in distancias_ordenadas[1:m+1] if v != i]
        
        for vecino in vecinos:
            ruta = hijo.copy()
            
            # Remove current city and reinsert it near the neighbor
            ruta.remove(i)
            insert_idx = ruta.index(vecino) + 1
            ruta.insert(insert_idx, i)
            
            # Calculate route cost
            costo_ruta = costo(ruta, distancias)
            
            if costo_ruta < mejor_costo:
                mejor_costo = costo_ruta
                mejor_ruta = ruta.copy()
        
    return mejor_ruta

@njit
def heuristica_abruptos(hijo, m, distancias):
    return encontrar_mejor_ruta(hijo, m, distancias)

# Numba-compatible selection of best individuals
@njit
def seleccionar_mejores_dos(individuos, aptitudes_individuos):
    # Find indices of two best individuals
    primer_mejor = 0
    segundo_mejor = 1 if aptitudes_individuos[1] < aptitudes_individuos[0] else 0
    
    for i in range(2, len(individuos)):
        if aptitudes_individuos[i] < aptitudes_individuos[primer_mejor]:
            segundo_mejor = primer_mejor
            primer_mejor = i
        elif aptitudes_individuos[i] < aptitudes_individuos[segundo_mejor]:
            segundo_mejor = i
    
    return [primer_mejor, segundo_mejor]

# Main genetic algorithm
@njit
def ejecutar_algoritmo(num_pob, num_gen, pm, m, num_competidores, hijos_crossover, distancias):
    # Determine number of variables
    num_var = len(distancias)
    
    # Create initial population
    poblacion = []
    for _ in range(num_pob):
        individuo = list(range(num_var))
        random.shuffle(individuo)
        poblacion.append(individuo)
    
    # Calculate initial fitness
    aptitudes = [costo(ind, distancias) for ind in poblacion]
    
    # Track best historical solution
    mejor_aptitud_historico = aptitudes[0]
    idx_mejor = 0
    for i in range(1, len(aptitudes)):
        if aptitudes[i] < mejor_aptitud_historico:
            mejor_aptitud_historico = aptitudes[i]
            idx_mejor = i
    mejor_individuo_historico = poblacion[idx_mejor]
    
    # Main evolution loop
    for _ in range(num_gen):
        # Parent selection via tournament
        padres = seleccion_torneo(poblacion, aptitudes, num_competidores)
        
        # Initialize children array
        hijos = []
        aptitudes_hijos = []
        
        # Crossover
        for i in range(0, num_pob, 2):
            if hijos_crossover == 1:
                # Single child case
                hijo1 = heuristica_abruptos(cycle_crossover(padres[i], padres[i+1]), m, distancias)
                aptitud_hijo1 = costo(hijo1, distancias)
                
                # Compare with parents and select best two
                individuos = [padres[i], padres[i+1], hijo1]
                aptitudes_individuos = [
                    costo(padres[i], distancias),
                    costo(padres[i+1], distancias),
                    aptitud_hijo1
                ]
                
                # Find indices of two best individuals
                mejores_indices = seleccionar_mejores_dos(individuos, aptitudes_individuos)
                
                # Add best individuals to offspring
                hijos.extend([individuos[j] for j in mejores_indices])
                aptitudes_hijos.extend([aptitudes_individuos[j] for j in mejores_indices])
            
            else:
                # Two children case
                hijo1 = heuristica_abruptos(cycle_crossover(padres[i], padres[i+1]), m, distancias)
                hijo2 = heuristica_abruptos(cycle_crossover(padres[i+1], padres[i]), m, distancias)
                
                aptitud_hijo1 = costo(hijo1, distancias)
                aptitud_hijo2 = costo(hijo2, distancias)
                
                # Compare with parents and select best two
                individuos = [padres[i], padres[i+1], hijo1, hijo2]
                aptitudes_individuos = [
                    costo(padres[i], distancias),
                    costo(padres[i+1], distancias),
                    aptitud_hijo1,
                    aptitud_hijo2
                ]
                
                # Find indices of two best individuals
                mejores_indices = seleccionar_mejores_dos(individuos, aptitudes_individuos)
                
                # Add best individuals to offspring
                hijos.extend([individuos[j] for j in mejores_indices])
                aptitudes_hijos.extend([aptitudes_individuos[j] for j in mejores_indices])
        
        # Update population and fitness
        poblacion = hijos
        aptitudes = aptitudes_hijos
        
        # Mutation
        poblacion, aptitudes = mutacion(poblacion, aptitudes, pm, distancias)
        
        # Update best historical solution
        mejor_aptitud_generacion = aptitudes[0]
        idx_mejor_generacion = 0
        for j in range(1, len(aptitudes)):
            if aptitudes[j] < mejor_aptitud_generacion:
                mejor_aptitud_generacion = aptitudes[j]
                idx_mejor_generacion = j
        
        if mejor_aptitud_generacion < mejor_aptitud_historico:
            mejor_aptitud_historico = mejor_aptitud_generacion
            mejor_individuo_historico = poblacion[idx_mejor_generacion]
    
    return mejor_aptitud_historico

# Execution of parameter combinations
def ejecutar_combinacion(param):
    distancias, combinacion, idx, total = param
    num_pob, num_gen, pm, m, num_competidores, hijos_crossover = combinacion

    iter_params = [
        (num_pob, num_gen, pm, m, num_competidores, hijos_crossover, distancias)
        for _ in range(10)
    ]

    # Measure execution time
    start_time = time.time()

    # Execute iterations
    resultados = [ejecutar_algoritmo(*params) for params in iter_params]

    # Calculate metrics
    mejor = min(resultados)
    promedio = sum(resultados) / len(resultados)
    peor = max(resultados)
    desviacion = statistics.stdev(resultados)
    tiempo_total = time.time() - start_time

    # Show progress
    print(f"{idx}/{total}")
    return [num_pob, num_gen, pm, m, num_competidores, hijos_crossover, 
            mejor, promedio, peor, desviacion, tiempo_total]

if __name__ == "__main__":
    # Load distance matrix
    try:
        # Open the file in read mode
        with open('Distancias_no_head.csv', 'r') as archivo:
            # Read all lines from the file
            lineas = archivo.readlines()

        # Create an empty matrix to store distances
        distancias = []

        # Iterate through each line
        for linea in lineas:
            # Split the line into elements using comma as delimiter
            fila = [float(valor.strip()) for valor in linea.split(',')]
            # Add the row to the distance matrix
            distancias.append(fila)

    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # Set parameter ranges to test
    param_ranges = [
        [20, 50, 100],     # Population size
        [50, 100, 200],    # Generations
        [0.05, 0.1, 0.2],  # Mutation probability
        [2, 3, 5],         # Heuristic parameter m
        [2, 3, 5],         # Tournament selection competitors
        [1, 2]             # Number of crossover children
    ]

    # Generate all parameter combinations
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

    # Execute parameter combinations
    for param in params:
        resultados_generales.append(ejecutar_combinacion(param))

    # Prepare results DataFrame
    columns = ["Num_Pob", "Num_Gen", "Pm", "m", "Num_Competidores", "Hijos_Crossover", 
               "Mejor", "Media", "Peor", "Desviacion", "Tiempo"]

    resultados_df = pd.DataFrame(resultados_generales, columns=columns)

    # Save to Excel
    output_file = "resultados_generales_Numba.xlsx"
    resultados_df.to_excel(output_file, index=False)

    print(f"Results saved to file: {output_file}")