#include "Biblioteca_c_limpio.h"

int main(int argc, char** argv){
    // Iniciar medición del tiempo
    time_t inicio = time(NULL);

    srand(time(NULL));
    int tamano_poblacion = 50;
    int longitud_genotipo = 32;
    int num_generaciones = 100;
    int num_competidores = 2;
    int m = 3;
    double probabilidad_mutacion = 0.15;
    double probabilidad_cruce = 0.99;
    char *nombre_archivo = "Distancias_no_head.csv";

    // Reservar memoria para la matriz
    double **distancias = malloc(longitud_genotipo * sizeof(double *));
    for (int i = 0; i < longitud_genotipo; i++) {
        distancias[i] = malloc(longitud_genotipo * sizeof(double));
    }

    // Abrir el archivo
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return 1;
    }

    // Leer el archivo y llenar la matriz
    char linea[8192];
    int fila = 0;
    while (fgets(linea, sizeof(linea), archivo) && fila < longitud_genotipo) {
        char *token = strtok(linea, ",");
        int columna = 0;
        while (token && columna < longitud_genotipo) {
            distancias[fila][columna] = atof(token);
            token = strtok(NULL, ",");
            columna++;
        }
        fila++;
        free(token);
    }
    fclose(archivo);

    // Crear población inicial
    poblacion *Poblacion = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *padres = crear_poblacion(tamano_poblacion, longitud_genotipo);    
    poblacion *hijos = crear_poblacion(tamano_poblacion, longitud_genotipo);
    crear_permutaciones(Poblacion, longitud_genotipo);
    evaluar_poblacion(Poblacion, distancias, longitud_genotipo);
    ordenar_poblacion(Poblacion);
    individuo *Mejor_Individuo = (individuo *)malloc(sizeof(individuo));
    Mejor_Individuo->genotipo = (int *)malloc(longitud_genotipo * sizeof(int));
    for (int i = 0; i < longitud_genotipo; i++) {
        Mejor_Individuo->genotipo[i] = Poblacion->individuos[0].genotipo[i];
    }
    Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;
    
    for(int generacion=0; generacion<num_generaciones; generacion++){
        // Seleccionar padres
        seleccionar_padres_torneo(Poblacion, padres, num_competidores, longitud_genotipo);

        // Cruzar padres
        cruzar_individuos(padres, hijos, tamano_poblacion, longitud_genotipo, m, distancias, probabilidad_cruce);

        // Mutar hijos
        for (int i = 0; i < tamano_poblacion; i++) {
            mutar_individuo(&hijos->individuos[i], distancias, probabilidad_mutacion, longitud_genotipo);
        }

        //Reemplazar la población
        actualizar_poblacion(&Poblacion, hijos, longitud_genotipo);

        // Evaluar hijos
        evaluar_poblacion(Poblacion, distancias, longitud_genotipo);
        ordenar_poblacion(Poblacion);

        // Actualizar el mejor individuo
        if (Poblacion->individuos[0].fitness < Mejor_Individuo->fitness) {
        for (int i = 0; i < longitud_genotipo; i++) {
            Mejor_Individuo->genotipo[i] = Poblacion->individuos[0].genotipo[i];
        }
        Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;
        }
    }
    // Imprimir el mejor individuo
    printf("Mejor Individuo: ");
    for (int i = 0; i < longitud_genotipo; i++) {
        printf("%d  ", Mejor_Individuo->genotipo[i]);
    }
    printf(" Fitness: %f\n", Mejor_Individuo->fitness);

    // Finalizar medición del tiempo
    time_t fin = time(NULL);
    double tiempo_ejecucion = difftime(fin, inicio);
    printf("Tiempo de ejecución: %.2f segundos\n", tiempo_ejecucion);

    // Liberar memoria
    liberar_poblacion(Poblacion);
    liberar_poblacion(padres);
    liberar_poblacion(hijos);
    for (int i = 0; i < longitud_genotipo; i++) {
        free(distancias[i]);
    }
    free(distancias);
    free(Mejor_Individuo->genotipo);
    Mejor_Individuo->genotipo = NULL;
    free(Mejor_Individuo);
    return 0;
}