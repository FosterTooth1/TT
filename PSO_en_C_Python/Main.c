#include "Biblioteca.h"

// Estructura para devolver resultados a Python
typedef struct {
    int *recorrido;
    double fitness;
    double tiempo_ejecucion;
    char (*nombres_ciudades)[50]; // Arreglo con los nombres de las ciudades correspondientes al mejor recorrido    
    int longitud_recorrido;
    double *fitness_generaciones; // Array de fitness por generación
} ResultadoPSO;

// Declaración de funciones exportadas
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

EXPORT ResultadoPSO* ejecutar_algoritmo_pso(int tamano_poblacion, int longitud_ruta, 
                                           int num_generaciones, float prob_pbest, 
                                           float prob_gbest, char* nombre_archivo,
                                           float prob_inercia) {

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

    // Creamos un arreglo con los nombres de las ciudades
    char nombres_ciudades[32][50] = {
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
    
    // Array para almacenar el fitness de cada generación
    double *fitness_generaciones = malloc(num_generaciones * sizeof(double));

    //Ejecutamos el PSO
    for (int generacion = 0; generacion < num_generaciones; generacion++) {
        // Actualizamos el cumulo
        actualizar_cumulo(cumulo, gbest, distancias, longitud_ruta, prob_pbest, prob_gbest, prob_inercia);

        // Ordenamos el cumulo de acuerdo a su fitness
        ordenar_cumulo(cumulo);

        // Guardamos el fitness de la generación actual
        fitness_generaciones[generacion] = fitness_gbest; // Guardamos el fitness de la generación actual

        // Actualizamos la mejor ruta global (gbest) si es necesario
        if (cumulo->particulas[0].fitness_mejor < fitness_gbest) {
            fitness_gbest = cumulo->particulas[0].fitness_mejor;
            memcpy(gbest, cumulo->particulas[0].mejor_ruta, longitud_ruta * sizeof(int));
        }
        
    }

    // Calculamos el tiempo total de ejecución del algoritmo
    time_t fin = time(NULL);
    double tiempo_ejecucion = difftime(fin, inicio);

    // Preparamos el resultado para devolverlo a Python
    ResultadoPSO* resultado = (ResultadoPSO*)malloc(sizeof(ResultadoPSO));
    resultado->recorrido = (int*)malloc(longitud_ruta * sizeof(int));
    resultado->fitness = fitness_gbest;
    resultado->tiempo_ejecucion = tiempo_ejecucion;
    resultado->nombres_ciudades = (char (*)[50])malloc(longitud_ruta * sizeof(char[50]));
    resultado->longitud_recorrido = longitud_ruta;
    resultado->fitness_generaciones = (double*)malloc(num_generaciones * sizeof(double));

    // Copiamos el recorrido y los nombres de las ciudades al resultado
    for (int i = 0; i < longitud_ruta; i++) {
        resultado->recorrido[i] = gbest[i];
        strcpy(resultado->nombres_ciudades[i], nombres_ciudades[gbest[i]]);
    }

    // Copiamos el fitness de cada generación al resultado
    for (int i = 0; i < num_generaciones; i++) {
        resultado->fitness_generaciones[i] = fitness_generaciones[i];
    }
    
    liberar_cumulo(cumulo);
    for (int i = 0; i < longitud_ruta; i++) {
        free(distancias[i]);
    }
    free(distancias);
    free(gbest);
    free(fitness_generaciones);

    return resultado;  // Devolvemos el resultado a Python
}

// Función para liberar la memoria del resultado en Python
EXPORT void liberar_resultado(ResultadoPSO* resultado) {
    if (resultado) {
        free(resultado->recorrido);
        free(resultado->nombres_ciudades);
        free(resultado->fitness_generaciones);
        free(resultado);
    }
}