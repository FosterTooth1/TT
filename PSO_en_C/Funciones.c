#include "Biblioteca.h"

// Asigna memoria para una cumulo
// Recibe el tamaño de el cumulo y la longitud de la ruta actual
// Devuelve un puntero a el cumulo creado
cumulo *crear_cumulo(int tamano, int longitud_ruta_actual)
{
    // Asigna memoria para la estructura de el cumulo
    cumulo *cumulo = malloc(sizeof(cumulo));

    // Asigna memoria para las particulas
    cumulo->tamano = tamano;
    cumulo->particulas = malloc(tamano * sizeof(particula));

    // Asigna memoria para la ruta actual y la mejor ruta de cada particula
    for (int i = 0; i < tamano; i++)
    {
        cumulo->particulas[i].ruta_actual = malloc(longitud_ruta_actual * sizeof(int));
        cumulo->particulas[i].mejor_ruta = malloc(longitud_ruta_actual * sizeof(int));
        // Inicializa el fitness en 0
        cumulo->particulas[i].fitness_actual = 0;
        cumulo->particulas[i].fitness_mejor = 0;
    }
    return cumulo;
}

// Crea permutaciones aleatorias para cada particula de el cumulo
// Recibe un puntero a el cumulo y la longitud de la ruta actual
// No devuelve nada (todo se hace por referencia)
void crear_permutaciones(cumulo *cumulo, int longitud_ruta_actual)
{
    for (int i = 0; i < cumulo->tamano; i++)
    {

        // Inicializa la ruta actual con valores ordenados
        for (int j = 0; j < longitud_ruta_actual; j++)
        {
            cumulo->particulas[i].ruta_actual[j] = j;
        }

        // Mezcla el ruta_actual utilizando el algoritmo de Fisher-Yates
        for (int j = longitud_ruta_actual - 1; j > 0; j--)
        {
            int k = rand() % (j + 1);
            int temp = cumulo->particulas[i].ruta_actual[j];
            cumulo->particulas[i].ruta_actual[j] = cumulo->particulas[i].ruta_actual[k];
            cumulo->particulas[i].ruta_actual[k] = temp;
        }
    }
}

// Actualiza la ruta y el fitness de el cumulo tomando en cuenta la mejor ruta global y la mejor ruta personal de cada particula
// Recibe un puntero a el cumulo, un puntero a la mejor particula (gbest), una matriz de distancias, la longitud de la ruta actual
// la probabilidad de pbest y la probabilidad de gbest
// No devuelve nada (todo se hace por referencia)
void actualizar_cumulo(cumulo *cumulo, int *gbest, double **distancias, int longitud_ruta_actual, float prob_pbest, float prob_gbest)
{

    // Recorre cada particula en el cumulo
    for (int i = 0; i < cumulo->tamano; i++)
    {
        // Actualiza la ruta y el fitness de la particula
        actualizar_particula(&cumulo->particulas[i], gbest, distancias, longitud_ruta_actual, prob_pbest, prob_gbest);
    }
}

// Actualiza la ruta y el fitness de una particula tomando en cuenta la mejor ruta global y la mejor ruta personal
// Recibe un puntero a la particula, un puntero a la mejor particula (gbest), una matriz de distancias, la longitud de la ruta actual
// la probabilidad de pbest y la probabilidad de gbest
// No devuelve nada (todo se hace por referencia)
void actualizar_particula(particula *particula, int *gbest, double **distancias,
                          int longitud_ruta_actual, float prob_pbest, float prob_gbest)
{
    // Paso 1: Clonar rutas originales
    int *ruta_original = clonar_ruta(particula->ruta_actual, longitud_ruta_actual);
    int *temp_pbest = clonar_ruta(particula->mejor_ruta, longitud_ruta_actual);
    int *temp_gbest = clonar_ruta(gbest, longitud_ruta_actual);

    // Paso 2: Generar todos los swaps candidatos (pbest y gbest)
    Swap *swaps = malloc(2 * longitud_ruta_actual * sizeof(Swap)); // Máximo 2 swaps por posición
    int swap_count = 0;

    // Recopilar swaps basados en diferencias con pbest y gbest
    for (int i = 0; i < longitud_ruta_actual; i++)
    {
        // Generar swap para pbest si hay discrepancia
        if (ruta_original[i] != temp_pbest[i])
        {
            for (int j = 0; j < longitud_ruta_actual; j++)
            {
                if (ruta_original[j] == temp_pbest[i])
                {
                    swaps[swap_count++] = (Swap){i, j, prob_pbest}; // Almacenar swap
                    break;
                }
            }
        }

        // Generar swap para gbest si hay discrepancia
        if (ruta_original[i] != temp_gbest[i])
        {
            for (int j = 0; j < longitud_ruta_actual; j++)
            {
                if (ruta_original[j] == temp_gbest[i])
                {
                    swaps[swap_count++] = (Swap){i, j, prob_gbest}; // Almacenar swap
                    break;
                }
            }
        }
    }

    // Paso 3: Seleccionar el mejor swap por posición (mayor probabilidad)
    Swap *swaps_seleccionados = malloc(longitud_ruta_actual * sizeof(Swap));
    bool *posicion_procesada = malloc(longitud_ruta_actual * sizeof(bool));
    memset(posicion_procesada, 0, longitud_ruta_actual * sizeof(bool)); // Inicializar a false

    for (int s = 0; s < swap_count; s++)
    {
        int pos = swaps[s].i;
        if (!posicion_procesada[pos])
        {
            swaps_seleccionados[pos] = swaps[s];
            posicion_procesada[pos] = true;
        }
        else
        {
            // Comparar probabilidades y quedarse con la mayor
            if (swaps[s].prob > swaps_seleccionados[pos].prob)
            {
                swaps_seleccionados[pos] = swaps[s];
            }
        }
    }

    // Paso 4: Aplicar swaps seleccionados con su probabilidad
    int *nueva_ruta = clonar_ruta(ruta_original, longitud_ruta_actual);
    for (int i = 0; i < longitud_ruta_actual; i++)
    {
        if (posicion_procesada[i])
        {
            Swap swap = swaps_seleccionados[i];
            if ((float)rand() / RAND_MAX <= swap.prob)
            {
                aplicar_swap(nueva_ruta, swap.i, swap.j);
            }
        }
    }

    // Paso 5: Aplicar heurística de remoción de abruptos
    heuristica_abruptos(nueva_ruta, longitud_ruta_actual, 3, distancias);

    // Paso 6: Actualizar fitness y rutas
    double nuevo_fitness = calcular_fitness(nueva_ruta, distancias, longitud_ruta_actual);
    if (nuevo_fitness < particula->fitness_mejor)
    {
        memcpy(particula->mejor_ruta, nueva_ruta, longitud_ruta_actual * sizeof(int));
        particula->fitness_mejor = nuevo_fitness;
    }
    memcpy(particula->ruta_actual, nueva_ruta, longitud_ruta_actual * sizeof(int));
    particula->fitness_actual = nuevo_fitness;

    // Liberar memoria
    free(ruta_original);
    free(temp_pbest);
    free(temp_gbest);
    free(swaps);
    free(swaps_seleccionados);
    free(posicion_procesada);
    free(nueva_ruta);
}

// Función auxiliar para clonar una ruta
int *clonar_ruta(int *original, int longitud)
{
    int *copia = malloc(longitud * sizeof(int));
    memcpy(copia, original, longitud * sizeof(int));
    return copia;
}

// Función auxiliar para calcular el fitness (distancia total)
double calcular_fitness(int *ruta, double **distancias, int longitud)
{
    double total = 0.0;
    for (int i = 0; i < longitud - 1; i++)
    {
        total += distancias[ruta[i]][ruta[i + 1]];
    }
    total += distancias[ruta[longitud - 1]][ruta[0]]; // Cerrar el ciclo
    return total;
}

// Función auxiliar para aplicar un swap
void aplicar_swap(int *ruta, int i, int j)
{
    int temp = ruta[i];
    ruta[i] = ruta[j];
    ruta[j] = temp;
}

// Función principal de ordenamiento para el cumulo
// Recibe un puntero a el cumulo
// No devuelve nada (todo se hace por referencia)
void ordenar_cumulo(cumulo *cumulo)
{
    // Obtenemos el tamaño de el cumulo
    int n = cumulo->tamano;

    // Si el cumulo igual o menor a 1, no se hace nada
    if (n <= 1)
        return;

    // Calculamos la profundidad máxima de recursión
    int profundidad_max = 2 * log2_suelo(n);

    // Llamamos a la función auxiliar de ordenamiento introspectivo
    introsort_util(cumulo->particulas, &profundidad_max, 0, n);
}

// Libera la memoria usada por un cumulo
// Recibe un puntero a el cumulo
// No devuelve nada (todo se hace por referencia)
void liberar_cumulo(cumulo *cumulo)
{
    // Verifica si el cumulo es nulo
    if (cumulo == NULL)
        return;

    // Libera la memoria de las rutas de cada particula
    if (cumulo->particulas != NULL)
    {
        for (int i = 0; i < cumulo->tamano; i++)
        {
            if (cumulo->particulas[i].ruta_actual != NULL)
            {
                free(cumulo->particulas[i].ruta_actual);
                cumulo->particulas[i].ruta_actual = NULL;
            }
            if (cumulo->particulas[i].mejor_ruta != NULL)
            {
                free(cumulo->particulas[i].mejor_ruta);
                cumulo->particulas[i].mejor_ruta = NULL;
            }
        }
        free(cumulo->particulas);
        cumulo->particulas = NULL;
    }

    // Libera la memoria del cumulo
    free(cumulo);
}

// Heurística para remover abruptos en la ruta intercambiando ciudades mal posicionadas
// Recibe un puntero a la ruta, el número de ciudades total (longitud del genotipo), el número de ciudades más cercanas a considerar y la matriz de distancias
// No devuelve nada (todo se hace por referencia)
void heuristica_abruptos(int* ruta, int num_ciudades, int m, double** distancias) {
    // Inicializamos memoria para un arreglo temporal para la manipulación de rutas
    int* ruta_temp = malloc(num_ciudades * sizeof(int));

    // Inicializamos meemoria para la estructura que sirve para ordenar distancias
    DistanciaOrdenada* dist_ordenadas = malloc(num_ciudades * sizeof(DistanciaOrdenada));

    // Para cada ciudad en la ruta
    for (int i = 0; i < num_ciudades; i++) {
        int ciudad_actual = ruta[i];
        
        // Se obtiene y ordenan las m ciudades más cercanas
        for (int j = 0; j < num_ciudades; j++) {
            dist_ordenadas[j].distancia = distancias[ciudad_actual][j];
            dist_ordenadas[j].indice = j;
        }
        qsort(dist_ordenadas, num_ciudades, sizeof(DistanciaOrdenada), comparar_distancias);

        // Encontramos la posición actual de la ciudad en la ruta
        int pos_actual = -1;
        for (int j = 0; j < num_ciudades; j++) {
            if (ruta[j] == ciudad_actual) {
                pos_actual = j;
                break;
            }
        }

        // Inicializamos el mejor costo con el costo actual
        double mejor_costo = calcular_fitness(ruta, distancias, num_ciudades);
        int mejor_posicion = pos_actual;
        int mejor_vecino = -1;

        // Probamos la inserción con las m ciudades más cercanas
        for (int j = 1; j <= m && j < num_ciudades; j++) {
            int ciudad_cercana = dist_ordenadas[j].indice;
            
            // Encontramos la posición de la ciudad cercana
            int pos_cercana = -1;
            for (int k = 0; k < num_ciudades; k++) {
                if (ruta[k] == ciudad_cercana) {
                    pos_cercana = k;
                    break;
                }
            }

            if (pos_cercana != -1) {
                // Probar inserción antes y después de la ciudad cercana
                for (int posicion_antes_o_despues = 0; posicion_antes_o_despues <= 1; posicion_antes_o_despues++) {
                    memcpy(ruta_temp, ruta, num_ciudades * sizeof(int));
                    
                    // Eliminar de posición actual
                    eliminar_de_posicion(ruta_temp, num_ciudades, pos_actual);
                    
                    // Insertar en nueva posición (antes o después de la ciudad cercana)
                    int nueva_pos = pos_cercana + posicion_antes_o_despues;
                    if (nueva_pos > pos_actual) nueva_pos--;
                    if (nueva_pos >= num_ciudades) nueva_pos = num_ciudades - 1;
                    insertar_en_posicion(ruta_temp, num_ciudades, ciudad_actual, nueva_pos);

                    // Evaluar el nuevo costo
                    double nuevo_costo = calcular_fitness(ruta_temp, distancias, num_ciudades);
                    
                    // Actualizar el mejor costo y posición de la ciudad actual si es necesario
                    if (nuevo_costo < mejor_costo) {
                        mejor_costo = nuevo_costo;
                        mejor_posicion = nueva_pos;
                        mejor_vecino = ciudad_cercana;
                    }
                }
            }
        }

        // Si se encontró un mejor vecino, actualizar la ruta
        if (mejor_vecino != -1 && mejor_posicion != pos_actual) {
            memcpy(ruta_temp, ruta, num_ciudades * sizeof(int));
            eliminar_de_posicion(ruta_temp, num_ciudades, pos_actual);
            insertar_en_posicion(ruta_temp, num_ciudades, ciudad_actual, mejor_posicion);
            memcpy(ruta, ruta_temp, num_ciudades * sizeof(int));
        }
    }

    // Liberamos memoria
    free(ruta_temp);
    free(dist_ordenadas);
}

// Funciones auxiliares de ordenamiento

// Implementación de ordenamiento introspectivo
// Recibe un array de una particula, la profundidad máxima de recursión, el índice de inicio y fin
// No devuelve nada (todo se hace por referencia)
void introsort_util(particula *arr, int *profundidad_max, int inicio, int fin) {
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
int particion(particula *arr, int bajo, int alto) {
    // Encontramos el índice del pivote usando la mediana de tres
    int medio = bajo + (alto - bajo) / 2;
    int indice_pivote = mediana_de_tres(arr, bajo, medio, alto);
    
    // Movemos el pivote seleccionado al final del rango para facilitar partición
    intercambiar_particulas(&arr[indice_pivote], &arr[alto]);

    // Guardamos el elemento del pivote para comparación
    particula pivote = arr[alto];

    // i indica la última posición donde los elementos son menores o iguales al pivote
    int i = bajo - 1;

    // Recorremos el rango desde `bajo` hasta `alto - 1` (excluyendo el pivote)
    for (int j = bajo; j < alto; j++) {
        // Si el elemento actual es menor o igual al pivote
        if (arr[j].fitness_actual <= pivote.fitness_actual) {
            i++; // Avanzamos `i` para marcar la posición de intercambio
            intercambiar_particulas(&arr[i], &arr[j]); // Intercambiamos el elemento menor al pivote
        }
    }

    // Finalmente, colocamos el pivote en su posición correcta
    intercambiar_particulas(&arr[i + 1], &arr[alto]);

    // Retornamos la posición del pivote
    return i + 1;
}

// Función para encontrar la mediana de tres elementos (usado en quicksort para mejorar el balanceo)
// Recibe un array de particulas y tres índices
// Devuelve el índice de la mediana
int mediana_de_tres(particula *arr, int a, int b, int c) {
    // Se realizan comparaciones lógicas para encontrar la mediana
    if (arr[a].fitness_actual <= arr[b].fitness_actual) {
        if (arr[b].fitness_actual <= arr[c].fitness_actual)
            return b;
        else if (arr[a].fitness_actual <= arr[c].fitness_actual)
            return c;
        else
            return a;
    } else {
        if (arr[a].fitness_actual <= arr[c].fitness_actual)
            return a;
        else if (arr[b].fitness_actual <= arr[c].fitness_actual)
            return c;
        else
            return b;
    }
}

// Función para intercambiar dos elementos
// Recibe dos punteros a particulas
// No devuelve nada (todo se hace por referencia)
void intercambiar_particulas(particula *a, particula *b) {
    particula temp = *a;
    *a = *b;
    *b = temp;
}

// Ordenamiento por inserción para arreglos pequeños
// Recibe un array de particulas, el índice izquierdo y derecho
// No devuelve nada (todo se hace por referencia)
void insertion_sort(particula *arr, int izquierda, int derecha) {
    // Recorremos el array de izquierda a derecha
    for (int i = izquierda + 1; i <= derecha; i++) {
        // Insertamos el elemento actual en la posición correcta
        particula clave = arr[i];
        int j = i - 1;
        
        // Movemos los elementos mayores que la clave a una posición adelante
        while (j >= izquierda && arr[j].fitness_actual > clave.fitness_actual) {
            arr[j + 1] = arr[j];
            j--;
        }

        // Insertamos la clave en la posición correcta
        arr[j + 1] = clave;
    }
}

// Heapsort para ordenar a las particulas por fitness
// Recibe un array de particulas y el tamaño del array
// No devuelve nada (todo se hace por referencia)
void heapsort(particula *arr, int n) {
    // Construimos el montón (heapify)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // Extraemos los elementos del montón uno por uno
    for (int i = n - 1; i > 0; i--) {
        particula temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        heapify(arr, i, 0);
    }
}

// Función auxiliar para heapsort
// Recibe un array de particulas, el tamaño del array y un índice
// No devuelve nada (todo se hace por referencia)
void heapify(particula *arr, int n, int i) {
    // Inicializamos el mayor como el indice actual
    int mayor = i;

    // Calculamos los indices de los hijos izquierdo y derecho
    int izquierda = 2 * i + 1;
    int derecha = 2 * i + 2;

    // Si el hijo izquierdo es mayor que el padre actualizamos el mayor
    if (izquierda < n && arr[izquierda].fitness_actual > arr[mayor].fitness_actual)
        mayor = izquierda;

    // Si el hijo derecho es mayor que el padre actualizamos el mayor
    if (derecha < n && arr[derecha].fitness_actual > arr[mayor].fitness_actual)
        mayor = derecha;

    // Si el mayor no es el padre, intercambiamos y aplicamos heapify al subárbol
    if (mayor != i) {
        particula temp = arr[i];
        arr[i] = arr[mayor];
        arr[mayor] = temp;
        heapify(arr, n, mayor);
    }
}

//Funciones auxiliares de manipulación de arreglos (Usadas en la heurística de remoción de abruptos)

// Función de comparación para qsort
// Recibe dos punteros a distancia ordenada
// Devuelve un entero que indica la relación entre las distancias
int comparar_distancias(const void* a, const void* b) {
    DistanciaOrdenada* da = (DistanciaOrdenada*)a;
    DistanciaOrdenada* db = (DistanciaOrdenada*)b;
    if (da->distancia < db->distancia) return -1;
    if (da->distancia > db->distancia) return 1;
    return 0;
}

// Función para insertar un elemento en una posición específica del array
// Recibe un puntero al array, la longitud del array, el elemento a insertar y la posición
// No devuelve nada (todo se hace por referencia)
void insertar_en_posicion(int* array, int longitud, int elemento, int posicion) {
    for (int i = longitud - 1; i > posicion; i--) {
        array[i] = array[i - 1];
    }
    array[posicion] = elemento;
}

// Función para eliminar un elemento de una posición específica
// Recibe un puntero al array, la longitud del array y la posición
// No devuelve nada (todo se hace por referencia)
void eliminar_de_posicion(int* array, int longitud, int posicion) {
    for (int i = posicion; i < longitud - 1; i++) {
        array[i] = array[i + 1];
    }
}