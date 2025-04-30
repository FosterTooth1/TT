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

// Función para liberar todos los recursos en caso de error
static void liberar_recursos(double **distancias, int filas_asignadas, 
                           Solucion *sol, Solucion *actual, Solucion *mejor, 
                           int *vecino, double *hist)
{
    if (distancias) {
        for (int i = 0; i < filas_asignadas; i++) {
            if (distancias[i]) free(distancias[i]);
        }
        free(distancias);
    }
    if (sol) liberar_solucion(sol);
    if (actual) liberar_solucion(actual);
    if (mejor) liberar_solucion(mejor);
    if (vecino) free(vecino);
    if (hist) free(hist);
}

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

    // Inicializar todos los punteros a NULL para garantizar una limpieza segura
    double **distancias = NULL;
    Solucion *sol = NULL, *actual = NULL, *mejor = NULL;
    int *vecino = NULL;
    double *hist = NULL;
    ResultadoRecocido *R = NULL;
    FILE *f = NULL;

    // Asignar memoria para la matriz de distancias
    distancias = malloc(longitud_ruta * sizeof(double *));
    if (!distancias)
    {
        perror("malloc distancias");
        return NULL;
    }
    
    // Inicializar todos los punteros de fila a NULL para facilitar limpieza
    int filas_asignadas = 0;
    for (int i = 0; i < longitud_ruta; i++)
    {
        distancias[i] = malloc(longitud_ruta * sizeof(double));
        if (!distancias[i])
        {
            perror("malloc distancias[i]");
            liberar_recursos(distancias, filas_asignadas, NULL, NULL, NULL, NULL, NULL);
            return NULL;
        }
        filas_asignadas++;
    }

    // Abrir el archivo CSV
    f = fopen(nombre_archivo, "r");
    if (!f)
    {
        perror("abrir CSV");
        liberar_recursos(distancias, longitud_ruta, NULL, NULL, NULL, NULL, NULL);
        return NULL;
    }

    // Leer el archivo CSV
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
    f = NULL;

    // 2) Nombres de las ciudades
    char nombres_arr[32][50] = {
        "Aguascalientes", "Baja California", "Baja California Sur", "Campeche",
        "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango", "Guanajuato",
        "Guerrero", "Hidalgo", "Jalisco", "Estado de Mexico", "Michoacan",
        "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla", "Queretaro",
        "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora", "Tabasco",
        "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan", "Zacatecas", "CDMX"};

    // 3) Preparar soluciones
    sol = crear_solucion(longitud_ruta, longitud_ruta);
    if (!sol) {
        liberar_recursos(distancias, longitud_ruta, NULL, NULL, NULL, NULL, NULL);
        return NULL;
    }
    
    crear_permutacion(sol, longitud_ruta);
    if (heuristica == 1)
    {
        heuristica_abruptos(sol->ruta, longitud_ruta, m, distancias);
    }

    actual = crear_solucion(longitud_ruta, longitud_ruta);
    if (!actual) {
        liberar_recursos(distancias, longitud_ruta, sol, NULL, NULL, NULL, NULL);
        return NULL;
    }
    
    mejor = crear_solucion(longitud_ruta, longitud_ruta);
    if (!mejor) {
        liberar_recursos(distancias, longitud_ruta, sol, actual, NULL, NULL, NULL);
        return NULL;
    }
    
    memcpy(actual->ruta, sol->ruta, longitud_ruta * sizeof(int));
    actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
    memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
    mejor->fitness = actual->fitness;

    vecino = malloc(longitud_ruta * sizeof(int));
    if (!vecino) {
        liberar_recursos(distancias, longitud_ruta, sol, actual, mejor, NULL, NULL);
        return NULL;
    }

    // 4) Calcular temperatura inicial (desviación típica de 100 muestras)
    double suma = 0, suma2 = 0;
    for (int i = 0; i < 100; i++)
    {
        generar_vecino(actual->ruta, vecino, longitud_ruta);
        if (heuristica == 1)
        {
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
    hist = calloc(num_generaciones, sizeof(double));
    if (!hist) {
        perror("calloc hist");
        liberar_recursos(distancias, longitud_ruta, sol, actual, mejor, vecino, NULL);
        return NULL;
    }

    int k;

    // 6) Bucle de recocido
    for (k = 1; k <= num_generaciones && T > temperatura_final; k++)
    {
        // Enfriamiento logarítmico de Béltsman
        T = T0 / log(k + 1.0);

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
                    memcpy(mejor->ruta, vecino, longitud_ruta * sizeof(int)), mejor->fitness = fv;
            }
            neigh++;
        }

        if (heuristica == 1)
            heuristica_abruptos(actual->ruta, longitud_ruta, m, distancias);

        actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
        hist[k - 1] = mejor->fitness; // recuerda que ahora k arranca en 1
    }

    // Si no se ha llegado a la última generación, rellenar el resto del histórico
    for (int i = k; i <= num_generaciones; i++) {
        hist[i-1] = mejor->fitness;
    }

    time_t fin = time(NULL);
    double t_total = difftime(fin, inicio);

    // 7) Empaquetar resultado
    R = malloc(sizeof(ResultadoRecocido));
    if (!R) {
        liberar_recursos(distancias, longitud_ruta, sol, actual, mejor, vecino, hist);
        return NULL;
    }
    
    R->longitud_recorrido = longitud_ruta;
    R->recorrido = malloc(longitud_ruta * sizeof(int));
    if (!R->recorrido) {
        free(R);
        liberar_recursos(distancias, longitud_ruta, sol, actual, mejor, vecino, hist);
        return NULL;
    }
    
    R->fitness = mejor->fitness;
    R->tiempo_ejecucion = t_total;
    R->fitness_generaciones = hist;
    
    R->nombres_ciudades = malloc(longitud_ruta * sizeof(*R->nombres_ciudades));
    if (!R->nombres_ciudades) {
        free(R->recorrido);
        free(R);
        liberar_recursos(distancias, longitud_ruta, sol, actual, mejor, vecino, hist);
        return NULL;
    }

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

    // Ya no liberamos hist porque ahora forma parte del resultado
    return R;
}

EXPORT void liberar_resultado_recocido(ResultadoRecocido *R)
{
    if (!R)
        return;

    free(R->recorrido);            // Liberar array de enteros
    free(R->nombres_ciudades);     // Liberar array de nombres
    free(R->fitness_generaciones); // Liberar array de doubles
    free(R);                       // Liberar la estructura principal
}