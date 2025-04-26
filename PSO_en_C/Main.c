#include "Biblioteca.h"

int main(int argc, char** argv){
    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);

    // Parámetros del PSO
    srand(time(NULL));
    int tamano_poblacion = 150;
    int longitud_ruta = 32;
    int num_generaciones  = 500;
    int m = 3;
    float prob_gbest = 0.7;
    float prob_pbest = 0.35;
    float prob_inercia = 0.3;
    
    // Nombre del archivo con las distancias
    char *nombre_archivo = "Distancias_no_head.csv";

    // Reservamos memoria para la matriz que almacena las distancias
    double **distancias = malloc(longitud_ruta * sizeof(double *));
    for (int i = 0; i < longitud_ruta; i++) {
        distancias[i] = malloc(longitud_ruta * sizeof(double));
    }

    // Abrimos el archivo
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return 1;
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
        "Zacatecas", "CDMX"
    };

    //Inicializamos el cumulo
    Cumulo *cumulo = crear_cumulo(tamano_poblacion, longitud_ruta);

    //Creamos permutaciones aleatorias para cada particula de el cumulo
    crear_permutaciones(cumulo, longitud_ruta);

    //Calculamos el fitness de cada particula y su mejor ruta encontrada hasta el momento (pbest)
    for (int i = 0; i < tamano_poblacion; i++) {
        cumulo->particulas[i].fitness_actual = calcular_fitness(cumulo->particulas[i].ruta_actual, distancias, longitud_ruta);
        memcpy(cumulo->particulas[i].mejor_ruta, cumulo->particulas[i].ruta_actual, longitud_ruta * sizeof(int));
        cumulo->particulas[i].fitness_mejor = cumulo->particulas[i].fitness_actual;
    }

    // Ordenamos el cumulo de acuerdo a su fitness
    ordenar_cumulo(cumulo);

    // Inicializamos la mejor ruta global (gbest) con la mejor ruta de la primera particula
    int *gbest = malloc(longitud_ruta * sizeof(int));
    memcpy(gbest, cumulo->particulas[0].mejor_ruta, longitud_ruta * sizeof(int));
    double fitness_gbest = cumulo->particulas[0].fitness_mejor;
    
    //Ejecutamos el PSO
    for (int generacion = 0; generacion < num_generaciones; generacion++) {
        // Actualizamos el cumulo
        actualizar_cumulo(cumulo, gbest, distancias, longitud_ruta, prob_pbest, prob_gbest, prob_inercia);

        // Ordenamos el cumulo de acuerdo a su fitness
        ordenar_cumulo(cumulo);

        // Actualizamos la mejor ruta global (gbest) si es necesario
        if (cumulo->particulas[0].fitness_mejor < fitness_gbest) {
            fitness_gbest = cumulo->particulas[0].fitness_mejor;
            memcpy(gbest, cumulo->particulas[0].mejor_ruta, longitud_ruta * sizeof(int));
        }
    }

    // Imprimimos la mejor ruta global (gbest) y su fitness
    printf("Mejor ruta global (gbest): ");
    for (int i = 0; i < longitud_ruta; i++) {
        printf("%d  ", gbest[i]);
    }
    printf(" Fitness: %f\n", fitness_gbest);

    //Imprimimos el recorrido
    printf("Recorrido: ");
    for (int i = 0; i < longitud_ruta; i++) {
        printf("%s -> ", nombres_ciudades[gbest[i]]);
    }
    printf("\n");

    // Liberamos la memomoria de todos los elementos
    liberar_cumulo(cumulo);
    free(gbest);
    gbest = NULL;
    for (int i = 0; i < longitud_ruta; i++) {
        free(distancias[i]);
    }
    free(distancias);

    // Finalizamos la medición del tiempo
    time_t fin = time(NULL);

    // Imprimimos el tiempo de ejecución
    double tiempo_ejecucion = difftime(fin, inicio);
    printf("Tiempo de ejecucion: %.2f segundos\n", tiempo_ejecucion);

    return 0;
}