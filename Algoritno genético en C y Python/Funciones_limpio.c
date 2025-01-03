#include "Biblioteca_c_limpio.h"

poblacion *crear_poblacion(int tamano, int longitud_genotipo) {
    poblacion *Poblacion = malloc(sizeof(poblacion));
    if (Poblacion == NULL) return NULL;
    
    Poblacion->tamano = tamano;
    Poblacion->individuos = malloc(tamano * sizeof(individuo));
    if (Poblacion->individuos == NULL) {
        free(Poblacion);
        return NULL;
    }
    
    for(int i = 0; i < tamano; i++) {
        Poblacion->individuos[i].genotipo = malloc(longitud_genotipo * sizeof(int));
        if (Poblacion->individuos[i].genotipo == NULL) {
            // Liberar toda la memoria asignada hasta ahora
            for(int j = 0; j < i; j++) {
                free(Poblacion->individuos[j].genotipo);
            }
            free(Poblacion->individuos);
            free(Poblacion);
            return NULL;
        }
        Poblacion->individuos[i].fitness = 0;
    }
    return Poblacion;
}

void actualizar_poblacion(poblacion **destino, poblacion *origen, int longitud_genotipo) {
    if (destino == NULL || origen == NULL) return;
    
    // Crear una población temporal nueva
    poblacion *nueva = crear_poblacion(origen->tamano, longitud_genotipo);
    if (nueva == NULL) return;
    
    // Copiar los datos
    for (int i = 0; i < origen->tamano; i++) {
        for (int j = 0; j < longitud_genotipo; j++) {
            nueva->individuos[i].genotipo[j] = origen->individuos[i].genotipo[j];
        }
        nueva->individuos[i].fitness = origen->individuos[i].fitness;
    }
    
    // Liberar la población antigua si existe
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
    
    // Asignar la nueva población
    *destino = nueva;
}

void liberar_poblacion(poblacion *pob) {
    if (pob == NULL) return;
    
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
    free(pob);
}

void crear_permutaciones(poblacion *poblacion, int longitud_genotipo) {
    for (int i = 0; i < poblacion->tamano; i++) {
        // Crear permutación aleatoria para el genotipo
        for (int j = 0; j < longitud_genotipo; j++) {
            poblacion->individuos[i].genotipo[j] = j;
        }

        // Mezclar el genotipo utilizando el algoritmo de Fisher-Yates
        for (int j = longitud_genotipo - 1; j > 0; j--) {
            int k = rand() % (j + 1);
            int temp = poblacion->individuos[i].genotipo[j];
            poblacion->individuos[i].genotipo[j] = poblacion->individuos[i].genotipo[k];
            poblacion->individuos[i].genotipo[k] = temp;
        }
    }
}

// Función para evaluar un individuo
double evaluar_individuo(int *genotipo, double **distancias, int longitud_genotipo) {
    double total_cost = 0.0;
    for (int i = 0; i < longitud_genotipo - 1; i++) {
        total_cost += distancias[genotipo[i]][genotipo[i + 1]];
    }
    total_cost += distancias[genotipo[longitud_genotipo - 1]][genotipo[0]];
    return total_cost;
}

// Función para evaluar una población
void evaluar_poblacion(poblacion *poblacion, double **distancias, int longitud_genotipo) {
    // Evaluar cada individuo de la población
    for (int i = 0; i < poblacion->tamano; i++) {
        poblacion->individuos[i].fitness = evaluar_individuo(
            poblacion->individuos[i].genotipo, distancias, longitud_genotipo);
    }
}

void mutar_individuo(individuo *individuo, double **distancias, double probabilidad_mutacion, int longitud_genotipo) {
    // Generar un número aleatorio y determinar si se realiza la mutación
    if ((double)rand() / RAND_MAX < probabilidad_mutacion) {
        // Generar dos índices aleatorios distintos
        int idx1 = rand() % longitud_genotipo;
        int idx2 = rand() % longitud_genotipo;
        while (idx1 == idx2) {
            idx2 = rand() % longitud_genotipo;
        }

        // Intercambiar los genes en las posiciones idx1 e idx2
        int temp = individuo->genotipo[idx1];
        individuo->genotipo[idx1] = individuo->genotipo[idx2];
        individuo->genotipo[idx2] = temp;

        // Recalcular el fitness del individuo usando la nueva evaluar_individuo
        individuo->fitness = evaluar_individuo(individuo->genotipo, distancias, longitud_genotipo);
    }
}

void cruzar_individuos(poblacion *padres, poblacion *hijos, int num_pob, int longitud_genotipo, int m, double **distancias, double probabilidad_cruce) {
    for (int i = 0; i < num_pob; i += 2) {
        // Decidir si se realiza el cruce basado en la probabilidad
        if ((double)rand() / RAND_MAX < probabilidad_cruce) {
            // Aplicar cycle crossover y heurística de mejora
            int *hijo1 = malloc(longitud_genotipo * sizeof(int));
            int *hijo2 = malloc(longitud_genotipo * sizeof(int));

            cycle_crossover(padres->individuos[i].genotipo, padres->individuos[i + 1].genotipo, hijo1, longitud_genotipo);
            cycle_crossover(padres->individuos[i + 1].genotipo, padres->individuos[i].genotipo, hijo2, longitud_genotipo);

            heuristica_abruptos(hijo1, longitud_genotipo, m, distancias);
            heuristica_abruptos(hijo2, longitud_genotipo, m, distancias);

            // Crear array temporal para almacenar los individuos
            int **individuos = malloc(4 * sizeof(int *));
            individuos[0] = padres->individuos[i].genotipo;
            individuos[1] = padres->individuos[i + 1].genotipo;
            individuos[2] = hijo1;
            individuos[3] = hijo2;

            individuo temp_hijos[4];
            for (int j = 0; j < 4; j++) {
                temp_hijos[j].genotipo = individuos[j];
                temp_hijos[j].fitness = evaluar_individuo(individuos[j], distancias, longitud_genotipo);
            }

            // Seleccionar los mejores dos individuos
            int mejores_indices[2] = {0, 1};
            for (int j = 2; j < 4; j++) {
                if (temp_hijos[j].fitness < temp_hijos[mejores_indices[0]].fitness) {
                    mejores_indices[1] = mejores_indices[0];
                    mejores_indices[0] = j;
                } else if (temp_hijos[j].fitness < temp_hijos[mejores_indices[1]].fitness) {
                    mejores_indices[1] = j;
                }
            }

            // Asignar los mejores individuos a los hijos
            for (int j = 0; j < longitud_genotipo; j++) {
                hijos->individuos[i].genotipo[j] = individuos[mejores_indices[0]][j];
                hijos->individuos[i + 1].genotipo[j] = individuos[mejores_indices[1]][j];
            }
            hijos->individuos[i].fitness = temp_hijos[mejores_indices[0]].fitness;
            hijos->individuos[i + 1].fitness = temp_hijos[mejores_indices[1]].fitness;

            // Liberar memoria de los hijos temporales
            free(hijo1);
            free(hijo2);
            free(individuos);
            
        } else {
            // Si no hay cruce, copiar los padres directamente a los hijos
            for (int j = 0; j < longitud_genotipo; j++) {
                hijos->individuos[i].genotipo[j] = padres->individuos[i].genotipo[j];
                hijos->individuos[i + 1].genotipo[j] = padres->individuos[i + 1].genotipo[j];
            }
            hijos->individuos[i].fitness = padres->individuos[i].fitness;
            hijos->individuos[i + 1].fitness = padres->individuos[i + 1].fitness;
        }
    }
}

// Función de comparación para qsort
int comparar_distancias(const void* a, const void* b) {
    DistanciaOrdenada* da = (DistanciaOrdenada*)a;
    DistanciaOrdenada* db = (DistanciaOrdenada*)b;
    if (da->distancia < db->distancia) return -1;
    if (da->distancia > db->distancia) return 1;
    return 0;
}

// Función para insertar un elemento en una posición específica del array
void insertar_en_posicion(int* array, int longitud, int elemento, int posicion) {
    for (int i = longitud - 1; i > posicion; i--) {
        array[i] = array[i - 1];
    }
    array[posicion] = elemento;
}

// Función para eliminar un elemento de una posición específica
void eliminar_de_posicion(int* array, int longitud, int posicion) {
    for (int i = posicion; i < longitud - 1; i++) {
        array[i] = array[i + 1];
    }
}

// Heurística optimizada de remoción de abruptos
void heuristica_abruptos(int* ruta, int longitud_genotipo, int m, double** distancias) {
    // Arreglo temporal para manipulación de rutas
    int* ruta_temp = malloc(longitud_genotipo * sizeof(int));

    // Estructura para ordenar distancias
    DistanciaOrdenada* dist_ordenadas = malloc(longitud_genotipo * sizeof(DistanciaOrdenada));

    // Para cada ciudad en la ruta
    for (int i = 0; i < longitud_genotipo; i++) {
        int ciudad_actual = ruta[i];
        
        // Obtener y ordenar las m ciudades más cercanas
        for (int j = 0; j < longitud_genotipo; j++) {
            dist_ordenadas[j].distancia = distancias[ciudad_actual][j];
            dist_ordenadas[j].indice = j;
        }
        qsort(dist_ordenadas, longitud_genotipo, sizeof(DistanciaOrdenada), comparar_distancias);

        // Encontrar posición actual
        int pos_actual = -1;
        for (int j = 0; j < longitud_genotipo; j++) {
            if (ruta[j] == ciudad_actual) {
                pos_actual = j;
                break;
            }
        }

        double mejor_costo = evaluar_individuo(ruta, distancias, longitud_genotipo);
        int mejor_posicion = pos_actual;
        int mejor_vecino = -1;

        // Probar inserción con las m ciudades más cercanas
        for (int j = 1; j <= m && j < longitud_genotipo; j++) {
            int ciudad_cercana = dist_ordenadas[j].indice;
            
            // Encontrar posición de la ciudad cercana
            int pos_cercana = -1;
            for (int k = 0; k < longitud_genotipo; k++) {
                if (ruta[k] == ciudad_cercana) {
                    pos_cercana = k;
                    break;
                }
            }

            if (pos_cercana != -1) {
                // Probar inserción antes y después de la ciudad cercana
                for (int offset = 0; offset <= 1; offset++) {
                    memcpy(ruta_temp, ruta, longitud_genotipo * sizeof(int));
                    
                    // Eliminar de posición actual
                    eliminar_de_posicion(ruta_temp, longitud_genotipo, pos_actual);
                    
                    // Insertar en nueva posición
                    int nueva_pos = pos_cercana + offset;
                    if (nueva_pos > pos_actual) nueva_pos--;
                    if (nueva_pos >= longitud_genotipo) nueva_pos = longitud_genotipo - 1;
                    
                    insertar_en_posicion(ruta_temp, longitud_genotipo, ciudad_actual, nueva_pos);
                    
                    double nuevo_costo = evaluar_individuo(ruta_temp, distancias, longitud_genotipo);
                    
                    if (nuevo_costo < mejor_costo) {
                        mejor_costo = nuevo_costo;
                        mejor_posicion = nueva_pos;
                        mejor_vecino = ciudad_cercana;
                    }
                }
            }
        }

        // Aplicar el mejor movimiento encontrado
        if (mejor_vecino != -1 && mejor_posicion != pos_actual) {
            memcpy(ruta_temp, ruta, longitud_genotipo * sizeof(int));
            eliminar_de_posicion(ruta_temp, longitud_genotipo, pos_actual);
            insertar_en_posicion(ruta_temp, longitud_genotipo, ciudad_actual, mejor_posicion);
            memcpy(ruta, ruta_temp, longitud_genotipo * sizeof(int));
        }
    }

    // Liberar memoria
    free(ruta_temp);
    free(dist_ordenadas);
}

// Cycle crossover
void cycle_crossover(int *padre1, int *padre2, int *hijo, int longitud_genotipo) {
    // Inicializar el hijo con -1 (marca de no visitado)
    for (int i = 0; i < longitud_genotipo; i++) {
        hijo[i] = -1;
    }

    // Array para marcar posiciones visitadas
    int *visitado = calloc(longitud_genotipo, sizeof(int));

    int ciclo = 0;
    int posiciones_restantes = longitud_genotipo;

    // Mientras queden posiciones por visitar
    while (posiciones_restantes > 0) {
        // Encontrar la primera posición no visitada
        int inicio = -1;
        for (int i = 0; i < longitud_genotipo; i++) {
            if (!visitado[i]) {
                inicio = i;
                break;
            }
        }

        ciclo++;
        int actual = inicio;

        // Seguir el ciclo hasta que se cierre
        while (1) {
            // Marcar como visitado
            visitado[actual] = 1;
            posiciones_restantes--;

            // Asignar valor según el número de ciclo
            hijo[actual] = (ciclo % 2 == 1) ? padre1[actual] : padre2[actual];

            // Encontrar la siguiente posición
            int valor_buscar = padre2[actual];
            int siguiente = -1;
            for (int i = 0; i < longitud_genotipo; i++) {
                if (padre1[i] == valor_buscar) {
                    siguiente = i;
                    break;
                }
            }

            // Si la siguiente posición ya fue visitada, terminar este ciclo
            if (visitado[siguiente]) {
                break;
            }
            actual = siguiente;
        }
    }
    free(visitado);
}


void seleccionar_padres_torneo(poblacion *Poblacion, poblacion *padres, int num_competidores, int longitud_genotipo) {
    int tamano_poblacion = Poblacion->tamano;
    int tamano_padres = padres->tamano;
    int *indices_torneo = malloc(num_competidores * sizeof(int));

    for (int i = 0; i < tamano_poblacion; i++) {
        // Seleccionar al azar los competidores del torneo
        for (int j = 0; j < num_competidores; j++) {
            indices_torneo[j] = rand() % tamano_poblacion;
        }

        // Encontrar el ganador del torneo (menor fitness)
        int indice_ganador = indices_torneo[0];
        double mejor_fitness = Poblacion->individuos[indices_torneo[0]].fitness;

        for (int j = 1; j < num_competidores; j++) {
            int indice_actual = indices_torneo[j];
            double fitness_actual = Poblacion->individuos[indice_actual].fitness;

            if (fitness_actual < mejor_fitness) {
                mejor_fitness = fitness_actual;
                indice_ganador = indice_actual;
            }
        }

        // Copiar el individuo ganador a la población de padres
        for (int j = 0; j < longitud_genotipo; j++) {
            padres->individuos[i].genotipo[j] = Poblacion->individuos[indice_ganador].genotipo[j];
        }

        // Copiar el fitness del ganador
        padres->individuos[i].fitness = Poblacion->individuos[indice_ganador].fitness;
    }

    // Liberar memoria usada para los índices
    free(indices_torneo);
}

// Función auxiliar para heapsort
void heapify(individuo *arr, int n, int i) {
    int mayor = i;
    int izquierda = 2 * i + 1;
    int derecha = 2 * i + 2;

    if (izquierda < n && arr[izquierda].fitness > arr[mayor].fitness)
        mayor = izquierda;

    if (derecha < n && arr[derecha].fitness > arr[mayor].fitness)
        mayor = derecha;

    if (mayor != i) {
        individuo temp = arr[i];
        arr[i] = arr[mayor];
        arr[mayor] = temp;
        heapify(arr, n, mayor);
    }
}

// Implementación de heapsort
void heapsort(individuo *arr, int n) {
    // Construir el montón
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // Extraer elementos del montón uno por uno
    for (int i = n - 1; i > 0; i--) {
        individuo temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        heapify(arr, i, 0);
    }
}

// Ordenamiento por inserción para arreglos pequeños
void insertion_sort(individuo *arr, int izquierda, int derecha) {
    for (int i = izquierda + 1; i <= derecha; i++) {
        individuo clave = arr[i];
        int j = i - 1;
        
        while (j >= izquierda && arr[j].fitness > clave.fitness) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = clave;
    }
}

// Función para intercambiar dos elementos
void intercambiar_individuos(individuo *a, individuo *b) {
    individuo temp = *a;
    *a = *b;
    *b = temp;
}

// Función para encontrar la mediana de tres elementos
int mediana_de_tres(individuo *arr, int a, int b, int c) {
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

// Partición de quicksort usando la mediana de tres como pivote
int particion(individuo *arr, int bajo, int alto) {
    int medio = bajo + (alto - bajo) / 2;
    int indice_pivote = mediana_de_tres(arr, bajo, medio, alto);
    intercambiar_individuos(&arr[indice_pivote], &arr[alto]);
    
    individuo pivote = arr[alto];
    int i = bajo - 1;

    for (int j = bajo; j < alto; j++) {
        if (arr[j].fitness <= pivote.fitness) {
            i++;
            intercambiar_individuos(&arr[i], &arr[j]);
        }
    }
    intercambiar_individuos(&arr[i + 1], &arr[alto]);
    return i + 1;
}

// Calcular el logaritmo base 2 de n
int log2_suelo(int n) {
    int log = 0;
    while (n > 1) {
        n >>= 1;
        log++;
    }
    return log;
}

// Implementación de ordenamiento introspectivo
void introsort_util(individuo *arr, int *profundidad_max, int inicio, int fin) {
    int tamano = fin - inicio;
    
    // Si el tamaño de la partición es pequeño, usar ordenamiento por inserción
    if (tamano < 16) {
        insertion_sort(arr, inicio, fin - 1);
        return;
    }
    
    // Si la profundidad máxima es cero, cambiar a heapsort
    if (*profundidad_max == 0) {
        heapsort(arr + inicio, tamano);
        return;
    }
    
    // En caso contrario, usar quicksort
    (*profundidad_max)--;
    int pivote = particion(arr, inicio, fin - 1);
    introsort_util(arr, profundidad_max, inicio, pivote);
    introsort_util(arr, profundidad_max, pivote + 1, fin);
}

// Función principal de ordenamiento para la población
void ordenar_poblacion(poblacion *poblacion) {
    int n = poblacion->tamano;
    if (n <= 1) return;
    
    // Calcular la profundidad máxima de recursión
    int profundidad_max = 2 * log2_suelo(n);
    
    // Llamar a la función auxiliar de ordenamiento introspectivo
    introsort_util(poblacion->individuos, &profundidad_max, 0, n);
}