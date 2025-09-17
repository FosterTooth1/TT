# Franco Calderas Sergio Alberto 6BV1
# Problema TSP con ventanas de tiempo (TSP-TW)
import random
import numpy as np
import math
import os
import csv
import json

###########################################################################################################################################################
# Archivo con coordenadas de Ejemplo para mostrar en el mapa
rutaBase = "C:\\Users\\Legion\\OneDrive\\Documentos\\ESCOM\\8° semestre\\Trabajo Terminal 2\\Vistas Mock Ups"

###########################################################################################################################################################
def reordenarIndividuo(individuo):
    inicio = individuo.index(ciudadInicial) # Posicion de la ciudad inicial en el individuo
    nuevo_orden = individuo[inicio:] + individuo[:inicio] # Reordenar colocando la ciudad inicial al comienzo
    return nuevo_orden

def calcularAptitud(individuo):
    longitud = len(individuo)
    individuo_Ordenado = reordenarIndividuo(individuo) # Reordenar para facilitar la suma de los tiempos
    
    # Calculo del tiempo de llegada a cada ciudad
    tiempo_EntreCiudades = 0
    vectorTiempos = [0] * longitud
    for i in range(longitud):
        ciudadActual = individuo_Ordenado[i]
        ciudadSiguiente = individuo_Ordenado[(i + 1) % longitud] 

        # Tiempo acumulado para llegar a la siguiente ciudad
        tiempoLlegada = tiempo_EntreCiudades + matrizCostos[ciudadActual - 1][ciudadSiguiente - 1]

        # Si aun no abre la siguiente ventana se espera a que abra
        if tiempoLlegada < ventanaTiempo[ciudadSiguiente - 1][0]:  
            tiempoLlegada = ventanaTiempo[ciudadSiguiente - 1][0]

        vectorTiempos[i] = max(ventanaTiempo[ciudadSiguiente - 1][0], tiempo_EntreCiudades + matrizCostos[ciudadActual - 1][ciudadSiguiente - 1])
        tiempo_EntreCiudades = tiempoLlegada  # Actualizar el tiempo acumulado

    # Restricciones de ventanas de tiempo
    restriciones_TW = []
    for i in range(longitud):
        ciudadActual = individuo_Ordenado[i]
        ciudadSiguiente = individuo_Ordenado[(i + 1) % longitud] 

        g = vectorTiempos[i] - ventanaTiempo[ciudadSiguiente - 1][1] # Comparar con el cierre de la ventana de tiempo
        restriciones_TW.append(max(0, g))

    # Penalizacion
    P = sum(g ** 2 for g in restriciones_TW)
    VFO = max(vectorTiempos) + 10 * P
    return VFO

def cruzamiento_Cycle_Crossover(padre1, padre2):
    longitud = len(padre1)
    hijo1 = [-1] * longitud # Inicializar a los hijos
    hijo2 = [-1] * longitud
    visitados = [-1] * longitud # Inicializar todas las posiciones como no visitadas
    ciclo = True # False para el ciclo 0, True para el ciclo 1

    while -1 in visitados: # Mientras que no se hayan visitados todas las posiciones
        inicio = visitados.index(-1) # Buscar la siguiente posicion no visitada
        indice_actual = inicio
        indices = []

        while True:
            indices.append(indice_actual)
            visitados[indice_actual] = 1 # Marcar como visitado
            valor = padre2[indice_actual]
            indice_actual = padre1.index(valor) # Buscar en padre1
            if indice_actual == inicio: # Si empieza a repetirse el ciclo
                ciclo = not ciclo # Intercambiar el ciclo
                break

        # Asignar valores a los hijos
        for i in indices:
            if ciclo == False: # Alternar segun el ciclo
                hijo1[i] = padre1[i]
                hijo2[i] = padre2[i]
            else:
                hijo1[i] = padre2[i]
                hijo2[i] = padre1[i]

    return hijo1, hijo2

def remocion_Abruptos(individuo):
    for i in range(numCiudades):
        # Seleccion entre las ciudades mas cercanas
        distancias = matrizCostos[i]
        idx_ordenados = sorted(range(len(distancias)), key=lambda x: distancias[x]) 
        ciudades_cercanas = idx_ordenados[1 : m + 1] # Seleccionar las m ciudades
        ciudad_cercana = random.choice(ciudades_cercanas) + 1

        # Posicion de insercion
        posiciones_insercion = [individuo.index(ciudad_cercana)]
        posiciones_insercion.append(posiciones_insercion[0] + 1)

        # Eliminar ciudad de su posicions
        original = individuo.copy()
        posicion_remover = original.index(i + 1)
        original.pop(posicion_remover)

        # Ajustar las posiciones de insercion
        posiciones_insercion = [p if p < posicion_remover else p - 1 for p in posiciones_insercion]

        # Crear rutas nuevas 
        ruta1 = original[:posiciones_insercion[0]] + [i + 1] + original[posiciones_insercion[0]:]
        ruta2 = original[:posiciones_insercion[1]] + [i + 1] + original[posiciones_insercion[1]:]

        # Calcular las aptitudes
        aptitud_original = calcularAptitud(individuo)
        aptitud_ruta1 = calcularAptitud(ruta1)
        aptitud_ruta2 = calcularAptitud(ruta2)

        # Seleccionar la mejor ruta
        rutas = [individuo, ruta1, ruta2]
        aptitudes = [aptitud_original, aptitud_ruta1, aptitud_ruta2]
        mejor_ruta = rutas[aptitudes.index(min(aptitudes))]

        # Actualizar el individuo
        individuo = mejor_ruta

    return individuo

###########################################################################################################################################################
# FUNCION PRINCIPAL DEL ALGORITMO GENETICO HIBRIDO
iteraciones = [] # Arreglo para guardar al mejor individuo de cada iteracion
def algoritmoGeneticoHibrido(poblacion, probaMutaci, maxGeneracion):
    for _ in range(maxGeneracion):
        ###################################################################################################################################################
        # CRUZAMIENTO POR CYCLE CROSSOVER
        familiaOrdenada = []
        for i in range(0, len(poblacion), 2):
            padre1 = poblacion[i][0] # Permutacion del padre 1
            aptitud_padre1 = poblacion[i][1]
            padre2 = poblacion[i + 1][0] # Permutacion del padre 2
            aptitud_padre2 = poblacion[i + 1][1]

            permutacion_hijo1, permutacion_hijo2 = cruzamiento_Cycle_Crossover(padre1, padre2) # Generar las permutaciones de los hijos 

            # Aplicar la heuristica de remocion de abruptos a los hijos
            permutacion_hijo1 = remocion_Abruptos(permutacion_hijo1) 
            permutacion_hijo2 = remocion_Abruptos(permutacion_hijo2)
            
            # Evaluar las aptitudes de los hijos
            aptitud_hijo1 = calcularAptitud(permutacion_hijo1) 
            aptitud_hijo2 = calcularAptitud(permutacion_hijo2)
            
            # Conformar la estructura de los individuos
            hijo1 = (permutacion_hijo1, aptitud_hijo1)            
            hijo2 = (permutacion_hijo2, aptitud_hijo2)

            individuos_Ordenados = sorted([(padre1, aptitud_padre1), (padre2, aptitud_padre2), hijo1, hijo2], key=lambda x: x[1])
            familiaOrdenada.append(individuos_Ordenados)
        ###################################################################################################################################################
        # SUSTITUCION
        nuevaPoblacion = []
        for i in range(len(familiaOrdenada)): # Pasar a los dos mejores individuos de la familia ordenada
            nuevaPoblacion.append((familiaOrdenada[i][0]))
            nuevaPoblacion.append((familiaOrdenada[i][1]))
        ###################################################################################################################################################
        # MUTACION
        randMutac = random.uniform(0, 1) # Generar un numero random para comparar con probaMutaci
        if randMutac <= probaMutaci:
            indiceAleatorio = random.randint(0, len(nuevaPoblacion) - 1) # Generar un indice aleatorio del individuo para sustituir por Mutacion
            permutacion_nuevoIndividuo = random.sample(ciudades, numCiudades) # Generar una nueva permutacion de las ciudades
            aptitud_nuevoIndividuo = calcularAptitud(permutacion_nuevoIndividuo) # Evaluar la aptitud de la nueva permutacion
            nuevaPoblacion[indiceAleatorio] = (permutacion_nuevoIndividuo, aptitud_nuevoIndividuo) # Sustituir al individuo por la nueva permutacion

        # Actualizacion de la poblacion
        poblacion = nuevaPoblacion
        ###################################################################################################################################################
    # OBTENER AL MEJOR INDIVIDUO DE LA ULTIMA GENERACION
    aptitudes = [individuo[1] for individuo in poblacion]
    individuoMenorAptitud = aptitudes.index(min(aptitudes))
    iteraciones.append(poblacion[individuoMenorAptitud])
    minimaAptitud = poblacion[individuoMenorAptitud][1]
    print(f"  La Mejor Ruta es: {poblacion[individuoMenorAptitud][0]}")
    print(f"  Su aptitud es: {round(minimaAptitud, 2)} horas")
    return poblacion[individuoMenorAptitud]

###########################################################################################################################################################
# PARAMETROS PARA LA FUNCION DE ALGORITMO GENETICO HIBRIDO
maxGeneracion = 20
numIndividuos = 50
probaMutaci = 0.05
numEjecuciones = 5
m = 3 # Numero de ciudades cercanas para la remocion de abruptos
ciudadInicial = 1 # NY
matrizCostos = [[0, 61.82 , 18.54 , 37.52 , 54.08 , 1.88 , 59.98 , 32.82 , 69.42 , 36.76 , 60.26],
                [61.82 , 0 , 50.84 , 33.62 , 7.5 , 59.88 , 2.76 , 28.84 , 7.78 , 28.14 , 5.8],
                [18.54 , 50.84 , 0 , 26.74 , 43.38 , 18.6 , 49.28 , 22 , 58.7 , 23.36 , 49.3],
                [37.52 , 33.62 , 26.74 , 0 , 26.16 , 35.56 , 32.06 , 4.8 , 41.5 , 3.26 , 32.08],
                [54.08 , 7.5 , 43.38 , 26.16 , 0 , 52.06 , 7.32 , 21.38 , 15.34 , 20.68 , 5.92],
                [1.88 , 59.88 , 18.6 , 35.56 , 52.06 , 0 , 57.96 , 30.86 , 67.38 , 34.8 , 58.3],
                [59.98 , 2.76 , 49.28 , 32.06 , 7.32 , 57.96 , 0 , 27.28 , 10.62 , 26.58 , 6.76],
                [32.82 , 28.84 , 22 , 4.8 , 21.38 , 30.86 , 27.28 , 0 , 36.72 , 4.02 , 27.3],
                [69.42 , 7.78 , 58.7 , 41.5 , 15.34 , 67.38 , 10.62 , 36.72 , 0 , 36.02 , 12.14],
                [36.76 , 28.14 , 23.36 , 3.26 , 20.68 , 34.8 , 26.58 , 4.02 , 36.02 , 0 , 26.6],
                [60.26 , 5.8 , 49.3 , 32.08 , 5.92 , 58.3 , 6.76 , 27.3 , 12.14 , 26.6 , 0]]
numCiudades = len(matrizCostos)

experimento = 1
if experimento == 1: # Con ventanas de tiempo para cada ciudad
    ventanaTiempo = [[-math.inf, math.inf], [50, 90], [15, 25], [30, 55], [15, 75], [5, 35], [150, 200], [25, 50], [65, 100], [120, 150], [30, 85]]
elif experimento == 2: # Sin ventanas de tiempo
    ventanaTiempo = [[-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf], 
                     [-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf], [-math.inf, math.inf]]

#############################################################################################################################################################
# GENERAR POBLACION INICIAL
poblacion = []
ciudades = list(range(1, numCiudades + 1))
for i in range(numIndividuos):
    individuo = random.sample(ciudades, numCiudades) # Generar una permutacion aleatoria de las ciudades
    individuo = remocion_Abruptos(individuo) # Aplicar la heuristica de remocion de abruptos
    aptitud = calcularAptitud(individuo) # Calcular la aptitud del individuo
    poblacion.append((individuo, aptitud))

#############################################################################################################################################################
# LLAMADA A LA FUNCION algoritmoGeneticoHibrido
for i in range(numEjecuciones):
    print(f"\nEjecucion {i + 1}")
    algoritmoGeneticoHibrido(poblacion, probaMutaci, maxGeneracion)
    
#############################################################################################################################################################
# VISUALIZACION DE RESULTADOS
aptitudesGenerales = [individuo[1] for individuo in iteraciones] # Obtener las aptitudes globales
mejorIndividuo = aptitudesGenerales.index(min(aptitudesGenerales)) # Obtener el indice del individuo con la menor aptitud
peorIndividuo = aptitudesGenerales.index(max(aptitudesGenerales)) # Obtener el indice del individuo con la peor aptitud

mejorAptitud = iteraciones[mejorIndividuo][1] # Mejor aptitud de todas las iteraciones
peorAptitud = iteraciones[peorIndividuo][1] # Peor aptitud de todas las iteraciones
promedioAptitudes = np.mean(aptitudesGenerales) # Promedio de las aptitudes
desvEst_Aptitudes = np.std(aptitudesGenerales) # Desviacion estandar de las aptitudes

# MOSTRAR LAS METRICAS
print()
print("="*60)
print("RESULTADOS DE LAS MEJORES SOLUCIONES")
ruta = iteraciones[mejorIndividuo][0]
print(f"  La Mejor Ruta es: {ruta}") # Mostrar la ruta mas optima
print(f"  Su aptitud es: {round(mejorAptitud, 2)} horas") # Mostrar al individuo con aptitud
print("="*60)
print("RESULTADOS ESTADISTICOS")
print(f"  Mejor: {round(mejorAptitud, 2)}")
print(f"  Promedio: {round(promedioAptitudes, 2)}")
print(f"  Peor: {round(peorAptitud, 2)}")
print(f"  Desviacion Estandar: {round(desvEst_Aptitudes, 2)}")
print("="*60)

###########################################################################################################################################################
# Leer coordenadas
archivo_coords = os.path.join(rutaBase, "Coordenadas.csv")
coordenadas = []
with open(archivo_coords, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        coordenadas.append([float(row["Latitud"]), float(row["Longitud"])])

# Ordenar coordenadas segun la ruta
ruta_coords = [coordenadas[i-1] for i in ruta] 

# Guardar en formato JSON
ruta_json = os.path.join(rutaBase, "ruta.json")
with open(ruta_json, "w") as f:
    json.dump(ruta_coords, f)

