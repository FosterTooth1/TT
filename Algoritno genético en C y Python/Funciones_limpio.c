#include "Biblioteca_c_limpio.h"

poblacion *crear_poblacion(int tamano, int longitud_genotipo){
    // Reservar memoria para la población
    poblacion *Poblacion = malloc(sizeof(poblacion));
    Poblacion->tamano = tamano;
    Poblacion->individuos = malloc(tamano * sizeof(individuo));
    // Reservar memoria para los individuos
    for(int i = 0; i < tamano; i++){
        Poblacion->individuos[i].genotipo = malloc(longitud_genotipo * sizeof(int));
        Poblacion->individuos[i].fitness = 0;
    }
    return Poblacion;
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

void actualizar_poblacion(poblacion **destino, poblacion *origen, int longitud_genotipo) {
    // Liberar memoria anterior si existe
    if (*destino != NULL) {
        liberar_poblacion(*destino);
    }
    
    // Crear nueva población
    *destino = crear_poblacion(origen->tamano, longitud_genotipo);
    
    // Copiar datos
    for (int i = 0; i < origen->tamano; i++) {
        memcpy((*destino)->individuos[i].genotipo, 
               origen->individuos[i].genotipo, 
               longitud_genotipo * sizeof(int));
        (*destino)->individuos[i].fitness = origen->individuos[i].fitness;
    }
}

void liberar_poblacion(poblacion *pob) {
    if (pob != NULL) {
        if (pob->individuos != NULL) {
            for (int i = 0; i < pob->tamano; i++) {
                free(pob->individuos[i].genotipo);
                pob->individuos[i].genotipo = NULL;
            }
            free(pob->individuos);
            pob->individuos = NULL;
        }
        free(pob);
    }
}

void imprimir_poblacion(poblacion *poblacion, int longitud_genotipo){
    for (int i = 0; i < poblacion->tamano; i++) {
        printf("Individuo %d: ", i);
        // Imprimir el genotipo del individuo
        for (int j = 0; j < longitud_genotipo; j++) {
            printf("%d ", poblacion->individuos[i].genotipo[j]);
        }
        // Imprimir el fitness del individuo
        printf(" Fitness: %f\n", poblacion->individuos[i].fitness);
    }
}


void evaluar_poblacion(poblacion *poblacion, double **distancias, int longitud_genotipo) {
    // Evaluar cada individuo de la población
    for (int i = 0; i < poblacion->tamano; i++) {
        evaluar_individuo(&poblacion->individuos[i], distancias, longitud_genotipo);
    }
}

void evaluar_individuo(individuo *individuo, double **distancias, int longitud_genotipo) {
    // Calcular el fitness del individuo
    double total_cost = 0.0;
    for (int i = 0; i < longitud_genotipo - 1; i++) {
        total_cost += distancias[individuo->genotipo[i]][individuo->genotipo[i + 1]];
    }
    total_cost += distancias[individuo->genotipo[longitud_genotipo - 1]][individuo->genotipo[0]];

    individuo->fitness = total_cost;
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

        // Recalcular el fitness del individuo
        evaluar_individuo(individuo, distancias, longitud_genotipo);
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
                evaluar_individuo(&temp_hijos[j], distancias, longitud_genotipo);
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

void copiar_arreglo(int *destino, int *origen, int longitud) {
    for (int i = 0; i < longitud; i++) {
        destino[i] = origen[i];
    }
}

/*
// Heurística de mejora para rutas
void heuristica_abruptos(int *ruta, int longitud_genotipo, int m, double **distancias) {
    // Verificar ruta inicial
    if (!verificar_si_es_permutacion(ruta, longitud_genotipo)) {
        printf("Error en ruta inicial de heurística abruptos:\n");
        imprimir_arreglo_debug(ruta, longitud_genotipo, "Ruta inicial");
        exit(1);
    }

    int *mejor_ruta = malloc(longitud_genotipo * sizeof(int));
    copiar_arreglo(mejor_ruta, ruta, longitud_genotipo);
    
    for (int i = 0; i < longitud_genotipo; i++) {
        // Crear array para almacenar las distancias y sus índices
        typedef struct {
            double distancia;
            int indice;
        } DistanciaIndice;
        
        DistanciaIndice *dist_ordenadas = malloc(longitud_genotipo * sizeof(DistanciaIndice));
        
        // Llenar el array con las distancias desde la ciudad i
        for (int j = 0; j < longitud_genotipo; j++) {
            dist_ordenadas[j].distancia = distancias[i][j];
            dist_ordenadas[j].indice = j;
        }
        
        // Ordenar las distancias (bubble sort simple para m elementos)
        for (int j = 0; j < m + 1; j++) {
            for (int k = 0; k < longitud_genotipo - 1 - j; k++) {
                if (dist_ordenadas[k].distancia > dist_ordenadas[k + 1].distancia) {
                    DistanciaIndice temp = dist_ordenadas[k];
                    dist_ordenadas[k] = dist_ordenadas[k + 1];
                    dist_ordenadas[k + 1] = temp;
                }
            }
        }
        
        // Calcular el costo actual
        individuo temp_individuo = { .genotipo = mejor_ruta, .fitness = 0.0 };
        evaluar_individuo(&temp_individuo, distancias, longitud_genotipo);
        double mejor_costo_actual = temp_individuo.fitness;
        
        // Probar intercambios con los m vecinos más cercanos
        for (int j = 1; j <= m && j < longitud_genotipo; j++) {  // Empezamos desde 1 para saltar la propia ciudad
            int vecino = dist_ordenadas[j].indice;
            
            // Encontrar las posiciones de las ciudades en la ruta
            int idx_ciudad = -1, idx_vecino = -1;
            for (int k = 0; k < longitud_genotipo; k++) {
                if (mejor_ruta[k] == i) idx_ciudad = k;
                if (mejor_ruta[k] == vecino) idx_vecino = k;
            }
            
            if (idx_ciudad != -1 && idx_vecino != -1) {
                // Crear ruta temporal y hacer el intercambio
                int *ruta_temp = malloc(longitud_genotipo * sizeof(int));
                copiar_arreglo(ruta_temp, mejor_ruta, longitud_genotipo);
                
                ruta_temp[idx_ciudad] = mejor_ruta[idx_vecino];
                ruta_temp[idx_vecino] = mejor_ruta[idx_ciudad];
                
                // Evaluar la nueva ruta
                temp_individuo.genotipo = ruta_temp;
                evaluar_individuo(&temp_individuo, distancias, longitud_genotipo);
                
                // Actualizar si mejora
                if (temp_individuo.fitness < mejor_costo_actual) {
                    copiar_arreglo(mejor_ruta, ruta_temp, longitud_genotipo);
                    mejor_costo_actual = temp_individuo.fitness;
                }
                
                free(ruta_temp);
            }
        }
        
        free(dist_ordenadas);
    }
    
    // Actualizar la ruta original con la mejor encontrada
    copiar_arreglo(ruta, mejor_ruta, longitud_genotipo);
    free(mejor_ruta);
    if (!verificar_si_es_permutacion(ruta, longitud_genotipo)) {
        printf("Error en ruta final de heurística abruptos:\n");
        imprimir_arreglo_debug(ruta, longitud_genotipo, "Ruta final");
        exit(1);
    }
}
*/

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

// Función para calcular el costo de una ruta
double calcular_costo_ruta(int* ruta, double** distancias, int longitud) {
    double costo = 0;
    for (int i = 0; i < longitud - 1; i++) {
        costo += distancias[ruta[i]][ruta[i + 1]];
    }
    costo += distancias[ruta[longitud - 1]][ruta[0]];
    return costo;
}

// Heurística optimizada de remoción de abruptos
void heuristica_abruptos(int* ruta, int longitud_genotipo, int m, double** distancias) {
    // Verificar validez de la ruta inicial
    if (!verificar_si_es_permutacion(ruta, longitud_genotipo)) {
        fprintf(stderr, "Error: Ruta inicial inválida en heurística\n");
        return;
    }

    // Arreglo temporal para manipulación de rutas
    int* ruta_temp = malloc(longitud_genotipo * sizeof(int));
    if (!ruta_temp) {
        fprintf(stderr, "Error: No se pudo asignar memoria para ruta temporal\n");
        return;
    }

    // Estructura para ordenar distancias
    DistanciaOrdenada* dist_ordenadas = malloc(longitud_genotipo * sizeof(DistanciaOrdenada));
    if (!dist_ordenadas) {
        free(ruta_temp);
        fprintf(stderr, "Error: No se pudo asignar memoria para distancias ordenadas\n");
        return;
    }

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

        double mejor_costo = calcular_costo_ruta(ruta, distancias, longitud_genotipo);
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
                    
                    double nuevo_costo = calcular_costo_ruta(ruta_temp, distancias, longitud_genotipo);
                    
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

    // Verificar validez de la ruta final
    if (!verificar_si_es_permutacion(ruta, longitud_genotipo)) {
        fprintf(stderr, "Error: Ruta final inválida en heurística\n");
    }
}

// Cycle crossover
void cycle_crossover(int *padre1, int *padre2, int *hijo, int longitud_genotipo) {
    // Inicializar el hijo con -1 (marca de no visitado)
    for (int i = 0; i < longitud_genotipo; i++) {
        hijo[i] = -1;
    }

    // Array para marcar posiciones visitadas
    int *visitado = calloc(longitud_genotipo, sizeof(int));
    if (!visitado) {
        fprintf(stderr, "Error: No se pudo asignar memoria para el array visitado\n");
        exit(1);
    }

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

    // Verificar que el resultado sea una permutación válida
    if (!verificar_si_es_permutacion(hijo, longitud_genotipo)) {
        printf("Error en hijo después del crossover:\n");
        imprimir_arreglo_debug(hijo, longitud_genotipo, "Hijo");
        imprimir_arreglo_debug(padre1, longitud_genotipo, "Padre1");
        imprimir_arreglo_debug(padre2, longitud_genotipo, "Padre2");
        exit(1);
    }
}


void seleccionar_padres_torneo(poblacion *Poblacion, poblacion *padres, int num_competidores, int longitud_genotipo) {
    // Verificar población inicial
    for (int i = 0; i < Poblacion->tamano; i++) {
        if (!verificar_si_es_permutacion(Poblacion->individuos[i].genotipo, longitud_genotipo)) {
            printf("Error en población inicial, individuo %d:\n", i);
            imprimir_arreglo_debug(Poblacion->individuos[i].genotipo, longitud_genotipo, "Individuo");
            exit(1);
        }
    }
    int tamano_poblacion = Poblacion->tamano;
    int tamano_padres = padres->tamano;
    int *indices_torneo = malloc(num_competidores * sizeof(int));
    
    // Asegurarse de que la población de padres tiene espacio suficiente
    if (tamano_padres != tamano_poblacion) {
        fprintf(stderr, "Error: La población de padres debe tener el mismo tamaño que la población inicial.\n");
        return;
    }

    if (!indices_torneo) {
        fprintf(stderr, "Error: No se pudo asignar memoria para los índices del torneo.\n");
        exit(EXIT_FAILURE);
    }

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

        // Copiar al individuo ganador a la población de padres
        padres->individuos[i] = Poblacion->individuos[indice_ganador];

    }

    // Liberar memoria usada para los índices
        free(indices_torneo);
    // Verificar población de padres
    for (int i = 0; i < tamano_poblacion; i++) {
        if (!verificar_si_es_permutacion(padres->individuos[i].genotipo, longitud_genotipo)) {
            printf("Error en población de padres, individuo %d:\n", i);
            imprimir_arreglo_debug(padres->individuos[i].genotipo, longitud_genotipo, "Individuo");
            exit(1);
        }
    }
}

// Función para verificar si un arreglo es una permutación válida
int verificar_si_es_permutacion(int *arreglo, int longitud) {
    // Crear un arreglo para marcar números encontrados
    int *encontrados = calloc(longitud, sizeof(int));
    if (!encontrados) {
        fprintf(stderr, "Error: No se pudo asignar memoria para verificación.\n");
        return 0;
    }
    
    // Verificar que cada número aparezca una vez y esté en el rango correcto
    for (int i = 0; i < longitud; i++) {
        // Verificar rango
        if (arreglo[i] < 0 || arreglo[i] >= longitud) {
            printf("Error: Número fuera de rango en posición %d: %d\n", i, arreglo[i]);
            free(encontrados);
            return 0;
        }
        
        // Verificar duplicados
        if (encontrados[arreglo[i]] == 1) {
            printf("Error: Número duplicado encontrado: %d\n", arreglo[i]);
            free(encontrados);
            return 0;
        }
        
        encontrados[arreglo[i]] = 1;
    }
    
    // Verificar que todos los números estén presentes
    for (int i = 0; i < longitud; i++) {
        if (encontrados[i] != 1) {
            printf("Error: Falta el número %d en la permutación\n", i);
            free(encontrados);
            return 0;
        }
    }
    
    free(encontrados);
    return 1;
}

// Función de ayuda para imprimir el arreglo cuando se encuentra un error
void imprimir_arreglo_debug(int *arreglo, int longitud, const char *mensaje) {
    printf("%s: ", mensaje);
    for (int i = 0; i < longitud; i++) {
        printf("%d ", arreglo[i]);
    }
    printf("\n");
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

// Función principal de ordenamiento para reemplazar ordenar_poblacion existente
void ordenar_poblacion(poblacion *poblacion) {
    int n = poblacion->tamano;
    if (n <= 1) return;
    
    // Calcular la profundidad máxima de recursión
    int profundidad_max = 2 * log2_suelo(n);
    
    // Llamar a la función auxiliar de ordenamiento introspectivo
    introsort_util(poblacion->individuos, &profundidad_max, 0, n);
}