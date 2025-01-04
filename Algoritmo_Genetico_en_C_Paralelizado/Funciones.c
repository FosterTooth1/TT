#include "Biblioteca_cuda.h"

// Función auxiliar para manejo de errores CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel para inicialización de estados random
__global__ void setup_curand_kernel(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Función para obtener el tamaño óptimo de bloque
void obtenerConfiguracionCUDA(int *blockSize, int *minGridSize, int *gridSize, int N) {
    cudaOccupancyMaxPotentialBlockSize(minGridSize, blockSize, evaluar_poblacion_kernel, 0, N);
    *gridSize = (N + *blockSize - 1) / *blockSize;
}

//Funciones principales del algoritmo genético

//Asigna memoria para una población
//Recibe el tamaño de la población y la longitud del genotipo
//Devuelve un puntero a la población creada
poblacion *crear_poblacion(int tamano, int longitud_genotipo) {
    poblacion *Poblacion = (poblacion *)malloc(sizeof(poblacion));
    if (Poblacion == NULL) {
        fprintf(stderr, "Error al asignar memoria para Poblacion\n");
        exit(EXIT_FAILURE);
    }

    Poblacion->tamano = tamano;
    Poblacion->individuos = (individuo *)malloc(tamano * sizeof(individuo));
    if (Poblacion->individuos == NULL) {
        fprintf(stderr, "Error al asignar memoria para individuos\n");
        free(Poblacion);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tamano; i++) {
        Poblacion->individuos[i].genotipo = (int *)malloc(longitud_genotipo * sizeof(int));
        if (Poblacion->individuos[i].genotipo == NULL) {
            fprintf(stderr, "Error al asignar memoria para genotipo del individuo %d\n", i);
            for (int j = 0; j < i; j++) {
                free(Poblacion->individuos[j].genotipo);
            }
            free(Poblacion->individuos);
            free(Poblacion);
            exit(EXIT_FAILURE);
        }
    }

    return Poblacion;
}

// Crea permutaciones aleatorias para cada individuo de la población
// Recibe un puntero a la población y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
void crear_permutaciones(poblacion *poblacion, int longitud_genotipo) {
    for (int i = 0; i < poblacion->tamano; i++) {
        
        // Inicializa el genotipo con valores ordenados
        for (int j = 0; j < longitud_genotipo; j++) {
            poblacion->individuos[i].genotipo[j] = j;
        }
        
        // Mezcla el genotipo utilizando el algoritmo de Fisher-Yates
        for (int j = longitud_genotipo - 1; j > 0; j--) {
            int k = rand() % (j + 1);
            int temp = poblacion->individuos[i].genotipo[j];
            poblacion->individuos[i].genotipo[j] = poblacion->individuos[i].genotipo[k];
            poblacion->individuos[i].genotipo[k] = temp;
        }
    }
}

// Evalua la población basándose en las distancias entre las ciudades (fitness)
// Recibe un puntero a la población, una matriz de distancias y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
// Kernel para evaluar toda la población
__global__ void evaluar_poblacion_kernel(individuo_gpu *poblacion, double *distancias, int tamano_poblacion, int longitud_genotipo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion) return;

    double total_cost = 0.0;
    int *genotipo = poblacion[idx].genotipo;

    for (int i = 0; i < longitud_genotipo - 1; i++) {
        total_cost += distancias[genotipo[i] * longitud_genotipo + genotipo[i + 1]];
    }
    total_cost += distancias[genotipo[longitud_genotipo - 1] * longitud_genotipo + genotipo[0]];
    
    poblacion[idx].fitness = total_cost;
}

// Función device para evaluar individuos
// Recibe un genotipo, una matriz de distancias y la longitud del genotipo
// Devuelve el fitness del individuo
__device__ double evaluar_individuo_gpu(int *ruta, double *distancias, int num_ciudades) {
    double fitness = 0.0;
    for (int i = 0; i < num_ciudades - 1; i++) {
        fitness += distancias[ruta[i] * num_ciudades + ruta[i + 1]];
    }
    fitness += distancias[ruta[num_ciudades - 1] * num_ciudades + ruta[0]];
    return fitness;
}

// Función principal de ordenamiento para la población
// Recibe un puntero a la población
// No devuelve nada (todo se hace por referencia)
void ordenar_poblacion(poblacion *poblacion) {
    // Obtenemos el tamaño de la población
    int n = poblacion->tamano;
    
    // Si la población igual o menor a 1, no se hace nada
    if (n <= 1) return;
    
    // Calculamos la profundidad máxima de recursión
    int profundidad_max = 2 * log2_suelo(n);
    
    // Llamamos a la función auxiliar de ordenamiento introspectivo
    introsort_util(poblacion->individuos, &profundidad_max, 0, n);
}

// Kernel para selección de padres por torneo
// Recibe un puntero a la población, un puntero a la población de padres, el número de competidores y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
__global__ void seleccionar_padres_kernel(individuo_gpu *poblacion, individuo_gpu *padres, int *indices_torneo, 
                                        int num_competidores, int tamano_poblacion, int longitud_genotipo, 
                                        curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion) return;

    // Generamos índices aleatorios para el torneo
    int indice_ganador = 0;
    double mejor_fitness = 1e9;

    for (int i = 0; i < num_competidores; i++) {
        int indice_actual = (int)(curand_uniform(&states[idx]) * tamano_poblacion);
        if (poblacion[indice_actual].fitness < mejor_fitness) {
            mejor_fitness = poblacion[indice_actual].fitness;
            indice_ganador = indice_actual;
        }
    }

    // Copiamos el ganador a la población de padres
    for (int i = 0; i < longitud_genotipo; i++) {
        padres[idx].genotipo[i] = poblacion[indice_ganador].genotipo[i];
    }
    padres[idx].fitness = poblacion[indice_ganador].fitness;
}

// Cruza a los padres para generar a los hijos dependiendo de una probabilidad de cruce
// Recibe un puntero a la población destino, un puntero a la población origen y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
// Kernel principal de cruzamiento
__global__ void cruzar_individuos_kernel(individuo_gpu *padres, individuo_gpu *hijos, double *distancias,
                                       double prob_cruce, int tamano_poblacion, int longitud_genotipo,
                                       int m, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion/2) return;

    int idx2 = idx * 2;
    
    if (curand_uniform(&states[idx]) < prob_cruce) {
        // Crear espacio para los hijos temporales
        int *hijo1 = new int[longitud_genotipo];
        int *hijo2 = new int[longitud_genotipo];
        
        // Realizar cycle crossover
        int *visitado = new int[longitud_genotipo]();
        int ciclo = 0;
        int posiciones_restantes = longitud_genotipo;

        // Inicializar hijos con -1
        for (int i = 0; i < longitud_genotipo; i++) {
            hijo1[i] = hijo2[i] = -1;
        }

        while (posiciones_restantes > 0) {
            int inicio = -1;
            for (int i = 0; i < longitud_genotipo; i++) {
                if (!visitado[i]) {
                    inicio = i;
                    break;
                }
            }

            ciclo++;
            int actual = inicio;

            while (true) {
                visitado[actual] = 1;
                posiciones_restantes--;

                hijo1[actual] = (ciclo % 2 == 1) ? padres[idx2].genotipo[actual] : padres[idx2 + 1].genotipo[actual];
                hijo2[actual] = (ciclo % 2 == 1) ? padres[idx2 + 1].genotipo[actual] : padres[idx2].genotipo[actual];

                int valor_buscar = (ciclo % 2 == 1) ? padres[idx2 + 1].genotipo[actual] : padres[idx2].genotipo[actual];
                int siguiente = -1;
                for (int i = 0; i < longitud_genotipo; i++) {
                    if (padres[idx2].genotipo[i] == valor_buscar) {
                        siguiente = i;
                        break;
                    }
                }

                if (visitado[siguiente]) break;
                actual = siguiente;
            }
        }

        // Aplicar heurística de abruptos a ambos hijos
        heuristica_abruptos_gpu(hijo1, longitud_genotipo, m, distancias);
        heuristica_abruptos_gpu(hijo2, longitud_genotipo, m, distancias);

        // Evaluar padres e hijos
        double fitness_padre1 = evaluar_individuo_gpu(padres[idx2].genotipo, distancias, longitud_genotipo);
        double fitness_padre2 = evaluar_individuo_gpu(padres[idx2 + 1].genotipo, distancias, longitud_genotipo);
        double fitness_hijo1 = evaluar_individuo_gpu(hijo1, distancias, longitud_genotipo);
        double fitness_hijo2 = evaluar_individuo_gpu(hijo2, distancias, longitud_genotipo);

        // Crear array temporal para selección
        double fitness_array[4] = {fitness_padre1, fitness_padre2, fitness_hijo1, fitness_hijo2};
        int *genotipos[4] = {padres[idx2].genotipo, padres[idx2 + 1].genotipo, hijo1, hijo2};

        // Seleccionar los dos mejores
        int mejores_indices[2] = {0, 1};
        for (int j = 2; j < 4; j++) {
            if (fitness_array[j] < fitness_array[mejores_indices[0]]) {
                mejores_indices[1] = mejores_indices[0];
                mejores_indices[0] = j;
            } else if (fitness_array[j] < fitness_array[mejores_indices[1]]) {
                mejores_indices[1] = j;
            }
        }

        // Asignar los mejores a los hijos
        for (int j = 0; j < longitud_genotipo; j++) {
            hijos[idx2].genotipo[j] = genotipos[mejores_indices[0]][j];
            hijos[idx2 + 1].genotipo[j] = genotipos[mejores_indices[1]][j];
        }
        hijos[idx2].fitness = fitness_array[mejores_indices[0]];
        hijos[idx2 + 1].fitness = fitness_array[mejores_indices[1]];

        // Liberar memoria
        delete[] hijo1;
        delete[] hijo2;
        delete[] visitado;
        
    } else {
        // Si no hay cruce, copiar padres directamente
        for (int i = 0; i < longitud_genotipo; i++) {
            hijos[idx2].genotipo[i] = padres[idx2].genotipo[i];
            hijos[idx2 + 1].genotipo[i] = padres[idx2 + 1].genotipo[i];
        }
        hijos[idx2].fitness = padres[idx2].fitness;
        hijos[idx2 + 1].fitness = padres[idx2 + 1].fitness;
    }
}

// Kernel para mutación
// Recibe un puntero a la población, una matriz de distancias, la probabilidad de mutación, el tamaño de la población, la longitud del genotipo y los estados random
// No devuelve nada (todo se hace por referencia)
__global__ void mutar_individuos_kernel(individuo_gpu *individuos, double *distancias,
                                      double prob_mutacion, int tamano_poblacion,
                                      int longitud_genotipo, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tamano_poblacion) return;

    if (curand_uniform(&states[idx]) < prob_mutacion) {
        int idx1 = (int)(curand_uniform(&states[idx]) * longitud_genotipo);
        int idx2 = (int)(curand_uniform(&states[idx]) * longitud_genotipo);
        
        while (idx1 == idx2) {
            idx2 = (int)(curand_uniform(&states[idx]) * longitud_genotipo);
        }

        // Intercambio de genes
        int temp = individuos[idx].genotipo[idx1];
        individuos[idx].genotipo[idx1] = individuos[idx].genotipo[idx2];
        individuos[idx].genotipo[idx2] = temp;

        // Recalcular fitness
        double total_cost = 0.0;
        for (int i = 0; i < longitud_genotipo - 1; i++) {
            total_cost += distancias[individuos[idx].genotipo[i] * longitud_genotipo + 
                                   individuos[idx].genotipo[i + 1]];
        }
        total_cost += distancias[individuos[idx].genotipo[longitud_genotipo - 1] * 
                               longitud_genotipo + individuos[idx].genotipo[0]];
        individuos[idx].fitness = total_cost;
    }
}

// Actualiza la población destino copiando los datos de la población origen
// Recibe un puntero a la población destino, un puntero a la población origen y la longitud del genotipo
// No devuelve nada (todo se hace por referencia)
void actualizar_poblacion(poblacion **destino, poblacion *origen, int longitud_genotipo) {
    // Crea una población temporal nueva
    poblacion *nueva = crear_poblacion(origen->tamano, longitud_genotipo);

    // Copia los datos
    for (int i = 0; i < origen->tamano; i++) {
        for (int j = 0; j < longitud_genotipo; j++) {
            nueva->individuos[i].genotipo[j] = origen->individuos[i].genotipo[j];
        }
        nueva->individuos[i].fitness = origen->individuos[i].fitness;
    }

    // Libera la población antigua si existe
    if (*destino != NULL) {
        for (int i = 0; i < (*destino)->tamano; i++) {
            if ((*destino)->individuos[i].genotipo != NULL) {
                free((*destino)->individuos[i].genotipo);
            }
        }
        if ((*destino)->individuos != NULL) {
            free((*destino)->individuos);
        }
        free(*destino);
    }

    // Asigna la nueva población
    *destino = nueva;
}

// Libera la memoria usada por una población
// Recibe un puntero a la población
// No devuelve nada (todo se hace por referencia)
void liberar_poblacion(poblacion *pob) {
    // Verifica si la población es nula
    if (pob == NULL) return;

    // Libera la memoria de los genotipos de cada individuo
    if (pob->individuos != NULL) {
        for (int i = 0; i < pob->tamano; i++) {
            if (pob->individuos[i].genotipo != NULL) {
                free(pob->individuos[i].genotipo);
                pob->individuos[i].genotipo = NULL;
            }
        }
        free(pob->individuos);
        pob->individuos = NULL;
    }

    // Libera la memoria de la población
    free(pob);
}

// Funciones auxiliares del cruzamiento

// Heurística para remover abruptos en la ruta intercambiando ciudades mal posicionadas
// Recibe un puntero a la ruta, el número de ciudades total (longitud del genotipo), el número de ciudades más cercanas a considerar y la matriz de distancias
// No devuelve nada (todo se hace por referencia)
// Heurística de abruptos para GPU
__device__ void heuristica_abruptos_gpu(int *ruta, int num_ciudades, int m, double *distancias) {
    // Memoria temporal para manipulación de rutas
    int *ruta_temp = new int[num_ciudades];
    DistanciaOrdenadaGPU *dist_ordenadas = new DistanciaOrdenadaGPU[num_ciudades];

    for (int i = 0; i < num_ciudades; i++) {
        int ciudad_actual = ruta[i];
        
        // Ordenar ciudades por distancia
        for (int j = 0; j < num_ciudades; j++) {
            dist_ordenadas[j].distancia = distancias[ciudad_actual * num_ciudades + j];
            dist_ordenadas[j].indice = j;
        }
        
        // Ordenamiento simple para GPU
        for (int j = 0; j < m; j++) {
            for (int k = j + 1; k < num_ciudades; k++) {
                if (comparar_distancias_gpu(dist_ordenadas[k], dist_ordenadas[j])) {
                    DistanciaOrdenadaGPU temp = dist_ordenadas[j];
                    dist_ordenadas[j] = dist_ordenadas[k];
                    dist_ordenadas[k] = temp;
                }
            }
        }

        int pos_actual = -1;
        for (int j = 0; j < num_ciudades; j++) {
            if (ruta[j] == ciudad_actual) {
                pos_actual = j;
                break;
            }
        }

        double mejor_costo = evaluar_individuo_gpu(ruta, distancias, num_ciudades);
        int mejor_posicion = pos_actual;
        int mejor_vecino = -1;

        for (int j = 1; j <= m && j < num_ciudades; j++) {
            int ciudad_cercana = dist_ordenadas[j].indice;
            
            int pos_cercana = -1;
            for (int k = 0; k < num_ciudades; k++) {
                if (ruta[k] == ciudad_cercana) {
                    pos_cercana = k;
                    break;
                }
            }

            if (pos_cercana != -1) {
                for (int posicion_antes_o_despues = 0; posicion_antes_o_despues <= 1; posicion_antes_o_despues++) {
                    // Copiar ruta actual
                    for (int k = 0; k < num_ciudades; k++) {
                        ruta_temp[k] = ruta[k];
                    }
                    
                    eliminar_de_posicion_gpu(ruta_temp, num_ciudades, pos_actual);
                    
                    int nueva_pos = pos_cercana + posicion_antes_o_despues;
                    if (nueva_pos > pos_actual) nueva_pos--;
                    if (nueva_pos >= num_ciudades) nueva_pos = num_ciudades - 1;
                    
                    insertar_en_posicion_gpu(ruta_temp, num_ciudades, ciudad_actual, nueva_pos);
                    
                    double nuevo_costo = evaluar_individuo_gpu(ruta_temp, distancias, num_ciudades);
                    
                    if (nuevo_costo < mejor_costo) {
                        mejor_costo = nuevo_costo;
                        mejor_posicion = nueva_pos;
                        mejor_vecino = ciudad_cercana;
                    }
                }
            }
        }

        if (mejor_vecino != -1 && mejor_posicion != pos_actual) {
            for (int k = 0; k < num_ciudades; k++) {
                ruta_temp[k] = ruta[k];
            }
            eliminar_de_posicion_gpu(ruta_temp, num_ciudades, pos_actual);
            insertar_en_posicion_gpu(ruta_temp, num_ciudades, ciudad_actual, mejor_posicion);
            for (int k = 0; k < num_ciudades; k++) {
                ruta[k] = ruta_temp[k];
            }
        }
    }

    delete[] ruta_temp;
    delete[] dist_ordenadas;
}

// Funciones auxiliares de ordenamiento

// Implementación de ordenamiento introspectivo
// Recibe un array de individuos, la profundidad máxima de recursión, el índice de inicio y fin
// No devuelve nada (todo se hace por referencia)
void introsort_util(individuo *arr, int *profundidad_max, int inicio, int fin) {
    // Calculamos el tamaño de la partición
    int tamano = fin - inicio;
    
    // Si el tamaño de la partición es pequeño, usamos el ordenamiento por inserción
    if (tamano < 16) {
        insertion_sort(arr, inicio, fin - 1);
        return;
    }
    
    // Si la profundidad máxima es cero, cambiamos a heapsort (para evitar peor caso de quicksort)
    if (*profundidad_max == 0) {
        heapsort(arr + inicio, tamano);
        return;
    }
    
    // En caso contrario, usamos quicksort
    (*profundidad_max)--;
    int pivote = particion(arr, inicio, fin - 1);
    introsort_util(arr, profundidad_max, inicio, pivote);
    introsort_util(arr, profundidad_max, pivote + 1, fin);
}

// Función para calcular el logaritmo en base 2 de un número entero (parte entera)
// Recibe un número entero
// Devuelve el logaritmo en base 2 (parte entera)
int log2_suelo(int n) {
    int log = 0;
    while (n > 1) {
        n >>= 1;
        log++;
    }
    return log;
}

// Partición de quicksort usando la mediana de tres como pivote
// Recibe un array de individuos, los índices bajo y alto
// Devuelve el índice del pivote
int particion(individuo *arr, int bajo, int alto) {
    // Encontramos el índice del pivote usando la mediana de tres
    int medio = bajo + (alto - bajo) / 2;
    int indice_pivote = mediana_de_tres(arr, bajo, medio, alto);
    
    // Movemos el pivote seleccionado al final del rango para facilitar partición
    intercambiar_individuos(&arr[indice_pivote], &arr[alto]);

    // Guardamos el elemento del pivote para comparación
    individuo pivote = arr[alto];

    // i indica la última posición donde los elementos son menores o iguales al pivote
    int i = bajo - 1;

    // Recorremos el rango desde `bajo` hasta `alto - 1` (excluyendo el pivote)
    for (int j = bajo; j < alto; j++) {
        // Si el elemento actual es menor o igual al pivote
        if (arr[j].fitness <= pivote.fitness) {
            i++; // Avanzamos `i` para marcar la posición de intercambio
            intercambiar_individuos(&arr[i], &arr[j]); // Intercambiamos el elemento menor al pivote
        }
    }

    // Finalmente, colocamos el pivote en su posición correcta
    intercambiar_individuos(&arr[i + 1], &arr[alto]);

    // Retornamos la posición del pivote
    return i + 1;
}

// Función para encontrar la mediana de tres elementos (usado en quicksort para mejorar el balanceo)
// Recibe un array de individuos y tres índices
// Devuelve el índice de la mediana
int mediana_de_tres(individuo *arr, int a, int b, int c) {
    // Se realizan comparaciones lógicas para encontrar la mediana
    if (arr[a].fitness <= arr[b].fitness) {
        if (arr[b].fitness <= arr[c].fitness)
            return b;
        else if (arr[a].fitness <= arr[c].fitness)
            return c;
        else
            return a;
    } else {
        if (arr[a].fitness <= arr[c].fitness)
            return a;
        else if (arr[b].fitness <= arr[c].fitness)
            return c;
        else
            return b;
    }
}

// Función para intercambiar dos elementos
// Recibe dos punteros a individuos
// No devuelve nada (todo se hace por referencia)
void intercambiar_individuos(individuo *a, individuo *b) {
    individuo temp = *a;
    *a = *b;
    *b = temp;
}

// Ordenamiento por inserción para arreglos pequeños
// Recibe un array de individuos, el índice izquierdo y derecho
// No devuelve nada (todo se hace por referencia)
void insertion_sort(individuo *arr, int izquierda, int derecha) {
    // Recorremos el array de izquierda a derecha
    for (int i = izquierda + 1; i <= derecha; i++) {
        // Insertamos el elemento actual en la posición correcta
        individuo clave = arr[i];
        int j = i - 1;
        
        // Movemos los elementos mayores que la clave a una posición adelante
        while (j >= izquierda && arr[j].fitness > clave.fitness) {
            arr[j + 1] = arr[j];
            j--;
        }

        // Insertamos la clave en la posición correcta
        arr[j + 1] = clave;
    }
}

// Heapsort para ordenar a los individuos por fitness
// Recibe un array de individuos y el tamaño del array
// No devuelve nada (todo se hace por referencia)
void heapsort(individuo *arr, int n) {
    // Construimos el montón (heapify)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // Extraemos los elementos del montón uno por uno
    for (int i = n - 1; i > 0; i--) {
        individuo temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        heapify(arr, i, 0);
    }
}

// Función auxiliar para heapsort
// Recibe un array de individuos, el tamaño del array y un índice
// No devuelve nada (todo se hace por referencia)
void heapify(individuo *arr, int n, int i) {
    // Inicializamos el mayor como el indice actual
    int mayor = i;

    // Calculamos los indices de los hijos izquierdo y derecho
    int izquierda = 2 * i + 1;
    int derecha = 2 * i + 2;

    // Si el hijo izquierdo es mayor que el padre actualizamos el mayor
    if (izquierda < n && arr[izquierda].fitness > arr[mayor].fitness)
        mayor = izquierda;

    // Si el hijo derecho es mayor que el padre actualizamos el mayor
    if (derecha < n && arr[derecha].fitness > arr[mayor].fitness)
        mayor = derecha;

    // Si el mayor no es el padre, intercambiamos y aplicamos heapify al subárbol
    if (mayor != i) {
        individuo temp = arr[i];
        arr[i] = arr[mayor];
        arr[mayor] = temp;
        heapify(arr, n, mayor);
    }
}

//Funciones auxiliares de manipulación de arreglos (Usadas en la heurística de remoción de abruptos)

// Función de comparación para qsort
// Recibe dos punteros a distancia ordenada
// Devuelve un entero que indica la relación entre las distancias
__device__ int comparar_distancias_gpu(DistanciaOrdenadaGPU a, DistanciaOrdenadaGPU b) {
    return a.distancia < b.distancia;
}

// Función para insertar un elemento en una posición específica del array
// Recibe un puntero al array, la longitud del array, el elemento a insertar y la posición
// No devuelve nada (todo se hace por referencia)
__device__ void insertar_en_posicion_gpu(int *ruta, int num_ciudades, int ciudad, int pos) {
    for (int i = num_ciudades - 1; i > pos; i--) {
        ruta[i] = ruta[i - 1];
    }
    ruta[pos] = ciudad;
}

// Función para eliminar un elemento de una posición específica
// Recibe un puntero al array, la longitud del array y la posición
// No devuelve nada (todo se hace por referencia)
// Funciones auxiliares para la heurística de abruptos
__device__ void eliminar_de_posicion_gpu(int *ruta, int num_ciudades, int pos) {
    int valor = ruta[pos];
    for (int i = pos; i < num_ciudades - 1; i++) {
        ruta[i] = ruta[i + 1];
    }
    ruta[num_ciudades - 1] = valor;
}