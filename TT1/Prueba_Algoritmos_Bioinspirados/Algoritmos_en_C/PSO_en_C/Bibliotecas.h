#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<stdbool.h>


// Estructuras

// Estructura para almacenar swaps candidatos
typedef struct {
    int i;          // Posición a modificar
    int j;          // Posición objetivo del swap
    float prob;     // Probabilidad asociada
} Swap;

// Estructura para una particula (Almacena la ruta actual y la mejor ruta encontrada hasta el momento)
typedef struct{
    int *ruta_actual;
    int *mejor_ruta;
    double fitness_actual;
    double fitness_mejor;
    Swap *swaps_anteriores;
    int num_swaps_anteriores;
}Particula;

// Estructura para un cumulo (Almacena un arreglo de particulas y su tamaño)
typedef struct{
    Particula *particulas;
    int tamano;
}Cumulo;

// Estructura para ordenar distancias (Almacena la distancia y el índice)(Usado en la heurística de remoción de abruptos)
typedef struct {
    double distancia;
    int indice;
} DistanciaOrdenada;

//Funciones principales del PSO
//Asigna memoria para un cumulo
Cumulo *crear_cumulo(int tamano, int longitud_permutacion);
//Crea permutaciones aleatorias para cada particula de el cumulo
void crear_permutaciones(Cumulo *cumulo, int longitud_permutacion);
//Actualizar la ruta y el fitness del cumulo tomando en cuenta la mejor ruta global y la mejor ruta personal de cada particula
void actualizar_cumulo(Cumulo *cumulo, int* gbest, double **distancias, int longitud_permutacion, float prob_pbest, float prob_gbest, float prob_inercia);
//Actualizar la ruta y el fitness de una particula tomando en cuenta la mejor ruta global y la mejor ruta personal
void actualizar_particula(Particula *particula, int* gbest, double **distancias, int longitud_permutacion, float prob_pbest, float prob_gbest, float prob_inercia);
//Ordena a la población de acuerdo a su fitness mediante el algoritmo de introsort
void ordenar_cumulo(Cumulo *cumulo);
//Libera la memoria usada para el cumulo
void liberar_cumulo(Cumulo *cumulo);

//Funciones auxiliares del cruzamiento
//La heurística se encarga de remover abruptos en la ruta intercamdiando ciudades mal posicionadas
void heuristica_abruptos(int *ruta, int num_ciudades, int m, double **distancias);

//Funciones auxiliares de ordenamiento
// Introsort es un algoritmo de ordenamiento híbrido que combina QuickSort, HeapSort e InsertionSort
void introsort_util(Particula *arr, int *profundidad_max, int inicio, int fin);
//Calcula el logaritmo base 2 de un número para medir la profundidad de recursividad que puede alcanzar QuickSort
int log2_suelo(int n);
//Particiona el arreglo para el QuickSort (Funcion auxiliar de Introsort en especifico para el QuickSort)
int particion(Particula *arr, int bajo, int alto);
//Calcula la mediana de tres elementos (Funcion auxiliar de Introsort en especifico para el QuickSort)
int mediana_de_tres(Particula *arr, int a, int b, int c);
//Intercambia dos particulas (Funcion auxiliar de Introsort en especifico para el QuickSort)
void intercambiar_particulas(Particula *a, Particula *b);
//Insertion sort es un algoritmo de ordenamiento simple y eficiente para arreglos pequeños
void insertion_sort(Particula *arr, int izquierda, int derecha);
//Heapsort es un algoritmo de ordenamiento basado en árboles binarios
void heapsort(Particula *arr, int n);
//Heapify es una función auxiliar para heapsort
void heapify(Particula *arr, int n, int i);

//Funciones auxiliares de manipulación de arreglos (Usadas en la heurística de remoción de abruptos)
//Compara dos distancias para ordenarlas
int comparar_distancias(const void* a, const void* b);
//Inserta un elemento en una posición específica del arreglo
void insertar_en_posicion(int* array, int longitud, int elemento, int posicion);
//Elimina un elemento de una posición específica del arreglo
void eliminar_de_posicion(int* array, int longitud, int posicion);

//Funciones auxiliares para la actualización de rutas y fitness
// Función auxiliar para clonar una ruta
int *clonar_ruta(int *original, int longitud);
// Función auxiliar para calcular el fitness (distancia total)
double calcular_fitness(int *ruta, double **distancias, int longitud);
// Función auxiliar para aplicar un swap
void aplicar_swap(int *ruta, int i, int j);