#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include<stdbool.h>


// Estructuras
// Estructura para una particula (Almacena la ruta actual y la mejor ruta encontrada hasta el momento)
typedef struct{
    int *ruta_actual;
    int *mejor_ruta;
    double fitness_actual;
    double fitness_mejor;
}particula;

// Estructura para un cumulo (Almacena un arreglo de particulas y su tamaño)
typedef struct{
    particula *particulas;
    int tamano;
}cumulo;


// Estructura para ordenar distancias (Almacena la distancia y el índice)(Usado en la heurística de remoción de abruptos)
typedef struct {
    double distancia;
    int indice;
} DistanciaOrdenada;

// Estructura para almacenar swaps candidatos
typedef struct {
    int i;          // Posición a modificar
    int j;          // Posición objetivo del swap
    float prob;     // Probabilidad asociada
} Swap;

//Funciones principales del PSO
//Asigna memoria para un cumulo
cumulo *crear_cumulo(int tamano, int longitud_permutacion);
//Crea permutaciones aleatorias para cada particula de el cumulo
void crear_permutaciones(cumulo *cumulo, int longitud_permutacion);
//Actualizar la ruta y el fitness del cumulo tomando en cuenta la mejor ruta global y la mejor ruta personal de cada particula
void actualizar_cumulo(cumulo *cumulo, int* gbest, double **distancias, int longitud_permutacion, float prob_pbest, float prob_gbest);
//Actualizar la ruta y el fitness de una particula tomando en cuenta la mejor ruta global y la mejor ruta personal
void actualizar_particula(particula *particula, int* gbest, double **distancias, int longitud_permutacion, float prob_pbest, float prob_gbest);
//Ordena a la población de acuerdo a su fitness mediante el algoritmo de introsort
void ordenar_cumulo(cumulo *cumulo);
//Libera la memoria usada para el cumulo
void liberar_cumulo(cumulo *cumulo);

//Funciones auxiliares del cruzamiento
//La heurística se encarga de remover abruptos en la ruta intercamdiando ciudades mal posicionadas
void heuristica_abruptos(int *ruta, int num_ciudades, int m, double **distancias);

//Funciones auxiliares de ordenamiento
// Introsort es un algoritmo de ordenamiento híbrido que combina QuickSort, HeapSort e InsertionSort
void introsort_util(particula *arr, int *profundidad_max, int inicio, int fin);
//Calcula el logaritmo base 2 de un número para medir la profundidad de recursividad que puede alcanzar QuickSort
int log2_suelo(int n);
//Particiona el arreglo para el QuickSort (Funcion auxiliar de Introsort en especifico para el QuickSort)
int particion(particula *arr, int bajo, int alto);
//Calcula la mediana de tres elementos (Funcion auxiliar de Introsort en especifico para el QuickSort)
int mediana_de_tres(particula *arr, int a, int b, int c);
//Intercambia dos particulas (Funcion auxiliar de Introsort en especifico para el QuickSort)
void intercambiar_particulas(particula *a, particula *b);
//Insertion sort es un algoritmo de ordenamiento simple y eficiente para arreglos pequeños
void insertion_sort(particula *arr, int izquierda, int derecha);
//Heapsort es un algoritmo de ordenamiento basado en árboles binarios
void heapsort(particula *arr, int n);
//Heapify es una función auxiliar para heapsort
void heapify(particula *arr, int n, int i);

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