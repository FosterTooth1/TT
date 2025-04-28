#include "Biblioteca.h"

int main()
{
    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);

    srand(time(NULL));
    int longitud_ruta = 32;
    int tenencia_tabu = 6;           // Número de iteraciones que un movimiento permanece tabú
    const int max_neighbours = 1000; // L(T) = k·N, con k entre 10 y 100; N= 32
    float umbral_est_global = 0.2; // Umbral de estancamiento global
    float umbral_est_local = 0.1;  // Umbral de estancamiento local
    int num_generaciones = 75;
    int m = 3;

    // Nombre del archivo con las distancias
    char *nombre_archivo = "Distancias_no_head.csv";

    // Reservamos memoria para la matriz que almacena las distancias
    double **distancias = malloc(longitud_ruta * sizeof(double *));
    for (int i = 0; i < longitud_ruta; i++)
    {
        distancias[i] = malloc(longitud_ruta * sizeof(double));
    }

    // Abrimos el archivo
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo)
    {
        perror("Error al abrir el archivo");
        return 1;
    }

    // Leemos el archivo y llenamos la matriz
    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), archivo) && fila < longitud_ruta)
    {
        char *token = strtok(linea, ",");
        int columna = 0;
        while (token && columna < longitud_ruta)
        {
            distancias[fila][columna] = atof(token);
            token = strtok(NULL, ",");
            columna++;
        }
        fila++;
    }
    fclose(archivo);

    // Creamos un arreglo con los nombres de las ciudades
    char nombres_ciudades[32][19] = {
        "Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de Mexico",
        "Michoacan", "Morelos", "Nayarit", "Nuevo Leon", "Oaxaca", "Puebla",
        "Queretaro", "Quintana Roo", "San Luis Potosi", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatan",
        "Zacatecas", "CDMX"};

    // Inicializamos la solucion
    Solucion *solucion = crear_solucion(longitud_ruta, longitud_ruta);

    // Creamos una permutacion aleatoria para la solucion
    crear_permutacion(solucion, longitud_ruta);

    // Aplicamos la heurística de remoción de abruptos
    heuristica_abruptos(solucion->ruta, longitud_ruta, m, distancias);

    // Inicialización del recocido
    Solucion *mejor = crear_solucion(longitud_ruta, longitud_ruta);
    Solucion *actual = crear_solucion(longitud_ruta, longitud_ruta);

    // Usar la solución heurística como inicial
    memcpy(actual->ruta, solucion->ruta, longitud_ruta * sizeof(int));
    actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);

    memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
    mejor->fitness = actual->fitness;

    int sin_mejora_global = 0;  // Contador de iteraciones sin mejora
    int sin_mejora_actual = 0;  // Contador de iteraciones sin mejora
    double mejor_global = mejor->fitness;  // Track del mejor fitness histórico
    double prev_actual_fitness = actual->fitness;  // inicializamos

    // Lista tabú dinámica
    Tabu *lista_tabu = NULL;
    int tamano_tabu = 0;
    int iteracion = 0;

        // Bucle principal del Tabu Search
        while (iteracion < num_generaciones)
        {
            int mejor_i, mejor_j;
            double mejor_fitness = INFINITY; // Inicializar con un valor mayor que el mejor fitness actual
            int *mejor_vecino = malloc(longitud_ruta * sizeof(int));
            bool encontrado = false;  // Flag para validar si encontramos vecino válido

            prev_actual_fitness = actual->fitness;

            // Generar y evaluar vecinos
            for (int v = 0; v < max_neighbours; v++)
            {
                int i, j;
                int *vecino = malloc(longitud_ruta * sizeof(int));

                // Generar movimiento 2-opt
                generar_vecino(actual->ruta, vecino, longitud_ruta, &i, &j);
                double fitness = calcular_fitness(vecino, distancias, longitud_ruta);

                // Verificar si el movimiento es tabú
                bool es_tabu = false;
                for (int t = 0; t < tamano_tabu; t++)
                {
                    if (lista_tabu[t].i == i && lista_tabu[t].j == j &&
                        (iteracion - lista_tabu[t].iteracion) < tenencia_tabu)
                    {
                        es_tabu = true;
                        break;
                    }
                }

                // Criterio de aspiración
            if (!es_tabu || (es_tabu && fitness < mejor_global))  // Comparar con mejor global
            {
                if (fitness < mejor_fitness)
                {
                    mejor_fitness = fitness;
                    memcpy(mejor_vecino, vecino, longitud_ruta * sizeof(int));
                    mejor_i = i;
                    mejor_j = j;
                    encontrado = true;
                }
            }
                free(vecino);
            }

            // Aplicar heurística de remoción de abruptos
            heuristica_abruptos(mejor_vecino, longitud_ruta, m, distancias);
            mejor_fitness = calcular_fitness(mejor_vecino, distancias, longitud_ruta);


            if (encontrado)
            {
                memcpy(actual->ruta, mejor_vecino, longitud_ruta * sizeof(int));
                actual->fitness = mejor_fitness;
    
                // Verificar mejora global
                if (actual->fitness < mejor_global) {
                    mejor_global = actual->fitness;
                    memcpy(mejor->ruta, actual->ruta, longitud_ruta * sizeof(int));
                    mejor->fitness = actual->fitness;
                    sin_mejora_global = 0;  // Resetear contador
                } else {
                    sin_mejora_global++;  // Incrementar contador de estancamiento
                }

                // Agregar movimiento a lista tabú
                lista_tabu = realloc(lista_tabu, (tamano_tabu + 1) * sizeof(Tabu));
                lista_tabu[tamano_tabu++] = (Tabu){mejor_i, mejor_j, iteracion};
            }
            free(mejor_vecino);

                // ——— Control de estancamiento local ———
            if (actual->fitness < prev_actual_fitness)
            {
                // ¡La solución actual mejoró, reiniciamos contadores!
                //sin_mejora_actual = 0;
            }
            else if (actual->fitness == prev_actual_fitness)
            {
                // La solución actual se estancó, contamos una iteración más
                sin_mejora_actual++;
            }
            else if (actual->fitness > prev_actual_fitness)
            {
                // Sigue atascada, contamos otra iteración más
                sin_mejora_actual++;
            }

            if (sin_mejora_global > umbral_est_global * num_generaciones && sin_mejora_actual > umbral_est_local * num_generaciones) {  // 15 iteraciones para 150 totales
                // Perturbación fuerte: 5 swaps aleatorios + reset lista tabú
                for (int p = 0; p < 5; p++) {
                    int i = rand() % longitud_ruta;
                    int j = rand() % longitud_ruta;
                    int temp = actual->ruta[i];
                    actual->ruta[i] = actual->ruta[j];
                    actual->ruta[j] = temp;
                }
                actual->fitness = calcular_fitness(actual->ruta, distancias, longitud_ruta);
                
                // Limpiar lista tabú para nueva exploración
                free(lista_tabu);
                lista_tabu = NULL;
                tamano_tabu = 0;
                
                sin_mejora_global = 0;
                sin_mejora_actual = 0;  // Reiniciar contadores
                printf("---- REINICIO ADAPTATIVO ----\n");
            }

            // Eliminar entradas expiradas de la lista tabú
            int nueva_tamano = 0;
            for (int t = 0; t < tamano_tabu; t++)
            {
                if ((iteracion - lista_tabu[t].iteracion) < tenencia_tabu)
                {
                    lista_tabu[nueva_tamano++] = lista_tabu[t];
                }
            }
            tamano_tabu = nueva_tamano;

            printf("Iter: %3d | Mejor: %.2f | Actual: %.2f | Tabu: %d\n",
                   iteracion, mejor->fitness, actual->fitness, tamano_tabu);
            iteracion++;
        }

    // Mostrar tiempo de ejecución
    time_t fin = time(NULL);
    double tiempo_ejecucion = difftime(fin, inicio);
    printf("Tiempo de ejecución: %.2f segundos\n", tiempo_ejecucion);

    // Liberar memoria y mostrar resultados
    free(lista_tabu);
    printf("\nMejor ruta encontrada (%.2f km):\n", mejor->fitness);
    for (int i = 0; i < longitud_ruta; i++)
    {
        printf("%s -> ", nombres_ciudades[mejor->ruta[i]]);
    }
    printf("%s\n", nombres_ciudades[mejor->ruta[0]]);

    liberar_solucion(solucion);
    liberar_solucion(actual);
    liberar_solucion(mejor);
    for (int i = 0; i < longitud_ruta; i++)
    {
        free(distancias[i]);
    }
    free(distancias);
    return 0;
}