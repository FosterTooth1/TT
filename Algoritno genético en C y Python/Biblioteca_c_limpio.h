#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>

// Estructuras
// Estructura para un individuon (Almacena el genotipo y el fitness)
typedef struct{
    int *genotipo;
    double fitness;
}individuo;

// Estructura para una población (Almacena un arreglo de individuos y su tamaño)
typedef struct{
    individuo *individuos;
    int tamano;
}poblacion;

// Estructura para ordenar distancias (Almacena la distancia y el índice)(Usado en la heurística de remoción de abruptos)
typedef struct {
    double distancia;
    int indice;
} DistanciaOrdenada;

//Funciones principales del algoritmo genético
poblacion *crear_poblacion(int tamano, int longitud_genotipo);
void crear_permutaciones(poblacion *poblacion, int longitud_genotipo); 
void evaluar_poblacion(poblacion *poblacion, double **distancias, int longitud_genotipo);
double evaluar_individuo(int *individuo, double **distancias, int longitud_genotipo);
void ordenar_poblacion(poblacion *poblacion);
void seleccionar_padres_torneo(poblacion *Poblacion, poblacion *padres, int num_competidores, int longitud_genotipo);
void cruzar_individuos(poblacion *padres, poblacion *hijos, int num_pob, int longitud_genotipo, int m, double **distancias, double probabilidad_cruce);
void mutar_individuo(individuo *individuo, double **distancias, double probabilidad_mutacion, int longitud_genotipo);
void actualizar_poblacion(poblacion **destino, poblacion *origen, int longitud_genotipo);
void liberar_poblacion(poblacion *poblacion);


//Funciones auxiliares del cruzamiento
void heuristica_abruptos(int *ruta, int longitud_genotipo, int m, double **distancias);
void cycle_crossover(int *padre1, int *padre2, int *hijo, int longitud_genotipo);

//Funciones auxiliares de ordenamiento
void heapify(individuo *arr, int n, int i);
void heapsort(individuo *arr, int n);
void insertion_sort(individuo *arr, int izquierda, int derecha);
void intercambiar_individuos(individuo *a, individuo *b);
int mediana_de_tres(individuo *arr, int a, int b, int c);
int particion(individuo *arr, int bajo, int alto);
int log2_suelo(int n);
void introsort_util(individuo *arr, int *profundidad_max, int inicio, int fin);

//Funciones auxiliares de manipulación de arreglos (Usadas en la heurística de remoción de abruptos)
int comparar_distancias(const void* a, const void* b);
void insertar_en_posicion(int* array, int longitud, int elemento, int posicion);
void eliminar_de_posicion(int* array, int longitud, int posicion);