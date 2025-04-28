#include "Biblioteca.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

typedef struct
{
    int *recorrido;
    double fitness;
    double tiempo_ejecucion;
    char (*nombres_ciudades)[50];
    int longitud_recorrido;
    double *fitness_generaciones;
} ResultadoRecocido;

EXPORT ResultadoRecocido *ejecutar_algoritmo_recocido(int longitud_ruta,
                                                      int num_generaciones,
                                                      double tasa_enfriamiento,
                                                      double temperatura_final,
                                                      int max_neighbours,
                                                      int m,
                                                      char *nombre_archivo,
                                                      int heuristica)
{
    time_t inicio = time(NULL);
    srand((unsigned)inicio);

    // 1) Carga distancias
    double **distancias = malloc(longitud_ruta * sizeof(double *));
    for (int i = 0; i < longitud_ruta; i++)
        distancias[i] = malloc(longitud_ruta * sizeof(double));

    FILE *f = fopen(nombre_archivo, "r");
    if (!f)
    {
        perror("abrir CSV");
        return NULL;
    }
    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), f) && fila < longitud_ruta)
    {
        char *tok = strtok(linea, ",");
        for (int col = 0; tok && col < longitud_ruta; col++)
        {
            distancias[fila][col] = atof(tok);
            tok = strtok(NULL, ",");
        }
        fila++;
    }
    fclose(f);

    // 2) Nombres de las ciudades
    char nombres_arr[32][50] = {
        "Aguascalientes", "Baja California", "Baja California Sur", "Campeche",
        "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango", "Guanajuato",
        "Guerrero", "Hidalgo", "Jalisco", "Estado de Mexico", "Michoacan",
        "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla", "Queretaro",
        "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora", "Tabasco",
        "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan", "Zacatecas", "CDMX"};

    // 3) Preparar soluciones
    Solucion *sol = crear_solucion(longitud_ruta, longitud_ruta);
    crear_permutacion(sol, longitud_ruta);
    if (heuristica == 1){
        heuristica_abruptos(sol->ruta, longitud_ruta, m, distancias);
    }
    
    Solucion *actual = crear_solucion(longitud_ruta, longitud_ruta);
    Solucion *mejor = crear_solucion(longitud_ruta, longitud_ruta);
    memcpy(actual->ruta, sol->ruta, longitud_ruta * sizeof(int));
    actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
    memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
    mejor->fitness = actual->fitness;

    int *vecino = malloc(longitud_ruta * sizeof(int));

    // 4) Calcular temperatura inicial (desviación típica de 100 muestras)
    double suma = 0, suma2 = 0;
    for (int i = 0; i < 100; i++)
    {
        generar_vecino(actual->ruta, vecino, longitud_ruta);
        if (heuristica == 1){
            heuristica_abruptos(vecino, longitud_ruta, m, distancias);     
        }
        double f = calcular_fitness(vecino, distancias, longitud_ruta);
        suma += f;
        suma2 += f * f;
    }
    double desv = sqrt((suma2 - suma * suma / 100) / 99);
    double T0 = desv;
    double T = T0;

    const int max_successes = (int)(0.5 * max_neighbours);

    // 5) Array para histórico de fitness
    double *hist = malloc(num_generaciones * sizeof(double));

    // 6) Bucle de recocido
    for (int k = 0; k < num_generaciones && T > temperatura_final; k++)
    {
        T = T0 / (k + 1); // Cauchy
        int neigh = 0, succ = 0;
        while (neigh < max_neighbours && succ < max_successes)
        {
            generar_vecino(actual->ruta, vecino, longitud_ruta);
            double fv = calcular_fitness(vecino, distancias, longitud_ruta);
            double p = probabilidad_aceptacion(actual->fitness, fv, T);
            if (p > ((double)rand() / RAND_MAX))
            {
                memcpy(actual->ruta, vecino, longitud_ruta * sizeof(int));
                actual->fitness = fv;
                succ++;
                if (fv < mejor->fitness)
                {
                    memcpy(mejor->ruta, vecino, longitud_ruta * sizeof(int));
                    mejor->fitness = fv;
                }
            }
            neigh++;
        }
        // heurística tras enfriar

        if (heuristica == 1){
            heuristica_abruptos(actual->ruta, longitud_ruta, m, distancias);
        }
        // Actualizar fitness
        actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
        hist[k] = mejor->fitness;
    }

    time_t fin = time(NULL);
    double t_total = difftime(fin, inicio);

    // 7) Empaquetar resultado
    ResultadoRecocido *R = malloc(sizeof(ResultadoRecocido));
    R->longitud_recorrido = longitud_ruta;
    R->recorrido = malloc(longitud_ruta * sizeof(int));
    R->fitness = mejor->fitness;
    R->tiempo_ejecucion = t_total;
    R->fitness_generaciones = hist;
    R->nombres_ciudades = malloc(longitud_ruta * sizeof(*R->nombres_ciudades));

    for (int i = 0; i < longitud_ruta; i++)
    {
        R->recorrido[i] = mejor->ruta[i];
        strcpy(R->nombres_ciudades[i], nombres_arr[R->recorrido[i]]);
    }

    // 8) Limpieza
    liberar_solucion(sol);
    liberar_solucion(actual);
    liberar_solucion(mejor);
    free(vecino);
    for (int i = 0; i < longitud_ruta; i++)
        free(distancias[i]);
    free(distancias);

    return R;
}

EXPORT void liberar_resultado_recocido(ResultadoRecocido *R)
{
    if (!R)
        return;
    free(R->recorrido);
    free(R->nombres_ciudades);
    free(R->fitness_generaciones);
    free(R);
}