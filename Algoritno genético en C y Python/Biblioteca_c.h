#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>
#include<stdbool.h>

typedef struct{
    int *genotipo;
    double fitness;
}individuo;

typedef struct{
    individuo *individuos;
    int tamano;
}poblacion;

poblacion *crear_poblacion(int tamano, int longitud_genotipo);
void crear_permutaciones(poblacion *poblacion, int longitud_genotipo); 
void liberar_poblacion(poblacion *poblacion);
void liberar_individuo(individuo *individuo);
void imprimir_poblacion(poblacion *poblacion, int longitud_genotipo);
void evaluar_poblacion(poblacion *poblacion, double **distancias, int longitud_genotipo);
void evaluar_individuo(individuo *individuo, double **distancias, int longitud_genotipo);
void mutar_individuo(individuo *individuo, double **distancias, double probabilidad_mutacion, int longitud_genotipo);
void cruzar_individuos(poblacion *padres, poblacion *hijos, int num_pob, int longitud_genotipo, int m, double **distancias, double probabilidad_cruce);
void copiar_arreglo(int *destino, int *origen, int longitud);
void heuristica_abruptos(int *ruta, int longitud_genotipo, int m, double **distancias);
void cycle_crossover(int *padre1, int *padre2, int *hijo, int longitud_genotipo);
void seleccionar_padres_torneo(poblacion *Poblacion, poblacion *padres, int num_competidores, int longitud_genotipo);
int verificar_si_es_permutacion(int *arreglo, int longitud);
void imprimir_arreglo_debug(int *arreglo, int longitud, const char *mensaje);
void ordenar_poblacion(poblacion *poblacion);
void heapify(individuo *arr, int n, int i);
void heapsort(individuo *arr, int n);
void insertion_sort(individuo *arr, int izquierda, int derecha);
void intercambiar_individuos(individuo *a, individuo *b);
int mediana_de_tres(individuo *arr, int a, int b, int c);
int particion(individuo *arr, int bajo, int alto);
int log2_suelo(int n);
void introsort_util(individuo *arr, int *profundidad_max, int inicio, int fin);
