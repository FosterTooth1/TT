#include "Bibliotecas.h"

// Estructura para almacenar el resultado del algoritmo de recocido simulado
typedef struct
{
    int *recorrido;
    double fitness;
    double tiempo_ejecucion;
    int longitud_recorrido;
    double *fitness_generaciones;
    double temperatura_inicial;
    double temperatura_final;
} ResultadoRecocido;

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// Función para ejecutar el algoritmo de recocido simulado
// Recibe los parámetros necesarios para la ejecución del algoritmo y el nombre del archivo con la matriz de distancias
EXPORT ResultadoRecocido *ejecutar_algoritmo_recocido(int longitud_ruta,
                                                      int num_generaciones,
                                                      double tasa_enfriamiento,
                                                      double temperatura_final,
                                                      int max_neighbours,
                                                      int m,
                                                      char *nombre_archivo,
                                                      int heuristica)
{
    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);

    // Reservamos memoria para la matriz que almacena las distancias
    double **distancias = malloc(longitud_ruta * sizeof(double *));
    for (int i = 0; i < longitud_ruta; i++) {
        distancias[i] = malloc(longitud_ruta * sizeof(double));
    }

    // Abrimos el archivo
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return NULL;
    }

    // Leemos el archivo y llenamos la matriz
    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), archivo) && fila < longitud_ruta) {
        char *token = strtok(linea, ",");
        int columna = 0;
        while (token && columna < longitud_ruta) {
            distancias[fila][columna] = atof(token);
            token = strtok(NULL, ",");
            columna++;
        }
        fila++;
        //free(token);
    }
    fclose(archivo);

    // Se crea la solución inicial
    Solucion *sol = crear_solucion(1, longitud_ruta);
    crear_permutacion(sol, longitud_ruta);

    // Calculamos su fitness
    sol->fitness = calcular_fitness(sol->ruta, distancias, longitud_ruta);

    // Se aplica la heurística si se ha seleccionado
    if (heuristica == 1)
    {
        heuristica_abruptos(sol->ruta, longitud_ruta, m, distancias);
        sol->fitness = calcular_fitness(sol->ruta, distancias, longitud_ruta);
    }

    // Se asigna memoria para la solución actual y la mejor solución
    Solucion *actual = crear_solucion(1, longitud_ruta);    
    Solucion *mejor = crear_solucion(1, longitud_ruta);

    // Copiamos la solución inicial a actual y mejor
    memcpy(actual->ruta, sol->ruta, longitud_ruta * sizeof(int));
    actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
    memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
    mejor->fitness = actual->fitness;

    // Se reserva memoria para el vecino
    int *vecino = malloc(longitud_ruta * sizeof(int));

    // Calculamos la temperatura inicial basada en la desviación estándar de 100 vecinos
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

    // Número máximo de éxitos por temperatura
    const int max_successes = (int)(0.5 * max_neighbours);

    // Se reserva memoria para el histórico de fitness por generación
    double *fitness_generaciones = (double *)malloc(num_generaciones * sizeof(double));

    // Bucle principal del recocido simulado
    int k;
    for (k = 1; k <= num_generaciones && T > temperatura_final; k++)
    {
        // Enfriamiento logarítmico de Béltsman
        // T = T0 / log(k + 1.0);

        // Enfriamiento geométrico
        T = T * tasa_enfriamiento;

        // Iteraciones por temperatura
        // Establecer los contadores de vecinos y éxitos
        int neigh = 0, succ = 0;

        // Mientras no se alcance el máximo de vecinos o éxitos
        while (neigh < max_neighbours && succ < max_successes)
        {
            // Generar un vecino
            generar_vecino(actual->ruta, vecino, longitud_ruta);
            // Calcular su fitness
            double fv = calcular_fitness(vecino, distancias, longitud_ruta);
            // Calcular la probabilidad de aceptación
            double p = probabilidad_aceptacion(actual->fitness, fv, T);
            // Decidir si se acepta el vecino
            if (p > ((double)rand() / RAND_MAX))
            {
                // Aceptar el vecino y actualizar la solución actual
                memcpy(actual->ruta, vecino, longitud_ruta * sizeof(int));
                actual->fitness = fv;
                // Incrementar el contador de éxitos
                succ++;
                // Actualizar la mejor solución si es necesario
                if (fv < mejor->fitness)
                    memcpy(mejor->ruta, vecino, longitud_ruta * sizeof(int)), mejor->fitness = fv;
            }
            // Incrementar el contador de vecinos
            neigh++;
        }
        // Aplicar la heurística de abruptos si se ha seleccionado
        if (heuristica == 1)
            heuristica_abruptos(actual->ruta, longitud_ruta, m, distancias);
        // Actualizar el fitness de la solución actual
        actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
        // Almacenar el fitness de la mejor solución en esta generación
        fitness_generaciones[k - 1] = mejor->fitness;
    }

    // Si no se ha llegado a la última generación, rellenar el resto del histórico
    for (int i = k; i <= num_generaciones; i++) {
        fitness_generaciones[i-1] = mejor->fitness;
    }

    // Medir el tiempo total de ejecución
    time_t fin = time(NULL);
    double t_total = difftime(fin, inicio);

    // Empqaquetamos el resultado
    ResultadoRecocido* R = (ResultadoRecocido*)malloc(sizeof(ResultadoRecocido));
    R->recorrido = (int*)malloc(longitud_ruta * sizeof(int));
    R->fitness = mejor->fitness;
    R->longitud_recorrido = longitud_ruta;
    R->tiempo_ejecucion = t_total;
    R->fitness_generaciones = fitness_generaciones;
    R->temperatura_inicial = T0;
    R->temperatura_final = T;

    for (int i = 0; i < longitud_ruta; i++) {
        R->recorrido[i] = mejor->ruta[i];
    }

    // Limpiar la memoria
    liberar_solucion(sol);
    liberar_solucion(actual);
    liberar_solucion(mejor);
    free(vecino);
    for (int i = 0; i < longitud_ruta; i++) {
        free(distancias[i]);
    }
    free(distancias);

    return R;
}

// Función para liberar la memoria del resultado del recocido simulado
EXPORT void liberar_resultado(ResultadoRecocido *R)
{
    if (R){
    free(R->recorrido);            // Liberar array de enteros
    free(R->fitness_generaciones); // Liberar array de doubles
    free(R);                       // Liberar la estructura principal
    }         
}