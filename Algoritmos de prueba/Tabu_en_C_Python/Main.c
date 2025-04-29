#include "Biblioteca.h"

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

typedef struct {
    int *recorrido;
    double fitness;
    double tiempo_ejecucion;
    char (*nombres_ciudades)[50];  // Array de strings de 50 caracteres
    int longitud_recorrido;
    double *fitness_generaciones;  // Array de valores double
} ResultadoTabu;

EXPORT ResultadoTabu *ejecutar_algoritmo_tabu(
    int longitud_ruta,
    int tenencia_tabu,
    int num_generaciones,
    int max_neighbours,
    float umbral_est_global,
    float umbral_est_local,
    int m,
    char *nombre_archivo,
    int heuristica)
{
    time_t inicio = time(NULL);
    srand(time(NULL));

    double **distancias = malloc(longitud_ruta * sizeof(double *));
    for (int i = 0; i < longitud_ruta; i++) {
        distancias[i] = malloc(longitud_ruta * sizeof(double));
    }

    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return NULL;
    }

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
    }
    fclose(archivo);

    char nombres_ciudades[32][50] = {
        "Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de Mexico",
        "Michoacan", "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla",
        "Queretaro", "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan",
        "Zacatecas", "CDMX"};

    Solucion *solucion = crear_solucion(longitud_ruta, longitud_ruta);
    crear_permutacion(solucion, longitud_ruta);
    if (heuristica == 1) {
        heuristica_abruptos(solucion->ruta, longitud_ruta, m, distancias);
    }

    Solucion *actual = crear_solucion(longitud_ruta, longitud_ruta);
    Solucion *mejor = crear_solucion(longitud_ruta, longitud_ruta);
    memcpy(actual->ruta, solucion->ruta, longitud_ruta * sizeof(int));
    actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
    memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
    mejor->fitness = actual->fitness;

    Tabu *lista_tabu = NULL;
    int tamano_tabu = 0;
    int iteracion = 0;
    int sin_mejora_global = 0;
    int sin_mejora_actual = 0;
    double mejor_global = mejor->fitness;
    double prev_actual_fitness = actual->fitness;

    double *fitness_generaciones = malloc(num_generaciones * sizeof(double));

    while (iteracion < num_generaciones) {
        int mejor_i, mejor_j;
        double mejor_fitness = INFINITY;
        int *mejor_vecino = malloc(longitud_ruta * sizeof(int));
        bool encontrado = false;

        prev_actual_fitness = actual->fitness;

        for (int v = 0; v < max_neighbours; v++) {
            int i, j;
            int *vecino = malloc(longitud_ruta * sizeof(int));
            generar_vecino(actual->ruta, vecino, longitud_ruta, &i, &j);
            double fitness = calcular_fitness(vecino, distancias, longitud_ruta);

            bool es_tabu = false;
            for (int t = 0; t < tamano_tabu; t++) {
                if (lista_tabu[t].i == i && lista_tabu[t].j == j &&
                    (iteracion - lista_tabu[t].iteracion) < tenencia_tabu) {
                    es_tabu = true;
                    break;
                }
            }

            if (!es_tabu || (es_tabu && fitness < mejor_global)) {
                if (fitness < mejor_fitness) {
                    mejor_fitness = fitness;
                    memcpy(mejor_vecino, vecino, longitud_ruta * sizeof(int));
                    mejor_i = i;
                    mejor_j = j;
                    encontrado = true;
                }
            }
            free(vecino);
        }

        if (encontrado) {
            if (heuristica == 1) {
                heuristica_abruptos(mejor_vecino, longitud_ruta, m, distancias);
                mejor_fitness = calcular_fitness(mejor_vecino, distancias, longitud_ruta);
            }
            memcpy(actual->ruta, mejor_vecino, longitud_ruta * sizeof(int));
            actual->fitness = mejor_fitness;

            if (actual->fitness < mejor_global) {
                mejor_global = actual->fitness;
                memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
                mejor->fitness = actual->fitness;
                sin_mejora_global = 0;
            } else {
                sin_mejora_global++;
            }

            lista_tabu = realloc(lista_tabu, (tamano_tabu + 1) * sizeof(Tabu));
            lista_tabu[tamano_tabu++] = (Tabu){mejor_i, mejor_j, iteracion};
        }
        free(mejor_vecino);

        if (actual->fitness >= prev_actual_fitness) {
            sin_mejora_actual++;
        } else {
            sin_mejora_actual = 0;
        }

        if (sin_mejora_global > (umbral_est_global * num_generaciones) &&
            sin_mejora_actual > (umbral_est_local * num_generaciones)) {
            for (int p = 0; p < 5; p++) {
                int i = rand() % longitud_ruta;
                int j = rand() % longitud_ruta;
                int temp = actual->ruta[i];
                actual->ruta[i] = actual->ruta[j];
                actual->ruta[j] = temp;
            }
            actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
            free(lista_tabu);
            lista_tabu = NULL;
            tamano_tabu = 0;
            sin_mejora_global = 0;
            sin_mejora_actual = 0;
        }

        int nueva_tamano = 0;
        for (int t = 0; t < tamano_tabu; t++) {
            if ((iteracion - lista_tabu[t].iteracion) < tenencia_tabu) {
                lista_tabu[nueva_tamano++] = lista_tabu[t];
            }
        }
        tamano_tabu = nueva_tamano;

        fitness_generaciones[iteracion] = mejor->fitness;
        iteracion++;
    }

    time_t fin = time(NULL);
    double tiempo_ejecucion = difftime(fin, inicio);

    ResultadoTabu *resultado = malloc(sizeof(ResultadoTabu));
    resultado->recorrido = malloc(longitud_ruta * sizeof(int));
    memcpy(resultado->recorrido, mejor->ruta, longitud_ruta * sizeof(int));
    resultado->fitness = mejor->fitness;
    resultado->tiempo_ejecucion = tiempo_ejecucion;
    resultado->longitud_recorrido = longitud_ruta;
    resultado->fitness_generaciones = fitness_generaciones;

    // En main.c, dentro de ejecutar_algoritmo_tabu:
    resultado->nombres_ciudades = malloc(longitud_ruta * sizeof(*resultado->nombres_ciudades));
    for (int i = 0; i < longitud_ruta; i++) {
        strncpy(resultado->nombres_ciudades[i], nombres_ciudades[mejor->ruta[i]], 49);  // Copiar máximo 49 caracteres
        resultado->nombres_ciudades[i][49] = '\0';  // Asegurar terminación nula
    }

    liberar_solucion(solucion);
    liberar_solucion(actual);
    liberar_solucion(mejor);
    free(lista_tabu);
    for (int i = 0; i < longitud_ruta; i++) {
        free(distancias[i]);
    }
    free(distancias);

    return resultado;
}

EXPORT void liberar_resultado_tabu(ResultadoTabu *resultado) {
    if (!resultado) return;
    free(resultado->recorrido);
    free(resultado->nombres_ciudades);
    free(resultado->fitness_generaciones);
    free(resultado);
}