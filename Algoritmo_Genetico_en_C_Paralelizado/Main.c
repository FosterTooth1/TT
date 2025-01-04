#include "Biblioteca_cuda.h"

// Macro para manejo de errores CUDA
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main(int argc, char** argv) {
    // Iniciamos la medición del tiempo
    time_t inicio = time(NULL);

    // Parámetros del algoritmo genético
    srand(time(NULL));
    int tamano_poblacion = 50;
    int longitud_genotipo = 32;
    int num_generaciones = 100;
    int num_competidores = 2;
    int m = 3;
    double probabilidad_mutacion = 0.15;
    double probabilidad_cruce = 0.99;

    // Nombre del archivo con las distancias
    const char *nombre_archivo = "Distancias_no_head.csv";  // Cambiado a const char*

    // Reservamos memoria para la matriz que almacena las distancias
    double **distancias = (double **)malloc(longitud_genotipo * sizeof(double *));  // Cast explícito añadido
    for (int i = 0; i < longitud_genotipo; i++) {
        distancias[i] = (double *)malloc(longitud_genotipo * sizeof(double));  // Cast explícito añadido
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
    while (fgets(linea, sizeof(linea), archivo) && fila < longitud_genotipo) {
        char *token = strtok(linea, ",");
        int columna = 0;
        while (token && columna < longitud_genotipo) {
            distancias[fila][columna] = atof(token);
            token = strtok(NULL, ",");
            columna++;
        }
        fila++;
    }
    fclose(archivo);

    // Creamos un arreglo con los nombres de las ciudades
    const char nombres_ciudades[32][20] = {  // Aumentado tamaño a 20 y cambiado a const
        "Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de México",
        "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla",
        "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán",
        "Zacatecas", "CDMX"
    };

    // Preparación de datos para GPU
    // Aplanar matriz de distancias para GPU
    double *h_distancias_flat = (double*)malloc(longitud_genotipo * longitud_genotipo * sizeof(double));
    for(int i = 0; i < longitud_genotipo; i++) {
        for(int j = 0; j < longitud_genotipo; j++) {
            h_distancias_flat[i * longitud_genotipo + j] = distancias[i][j];
        }
    }

    // Alocar memoria en GPU
    double *d_distancias;
    curandState *d_states;
    gpuErrchk(cudaMalloc(&d_distancias, longitud_genotipo * longitud_genotipo * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_states, tamano_poblacion * sizeof(curandState)));

    // Copiar distancias a GPU
    gpuErrchk(cudaMemcpy(d_distancias, h_distancias_flat, 
              longitud_genotipo * longitud_genotipo * sizeof(double), 
              cudaMemcpyHostToDevice));

     // Configuración de bloques
    int blockSize;
    int minGridSize;
    int gridSize;
    obtenerConfiguracionCUDA(&blockSize, &minGridSize, &gridSize, tamano_poblacion);

    // Inicializar estados random en GPU
    setup_curand_kernel<<<gridSize, blockSize>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // Inicializamos las poblaciones en CPU
    poblacion *Poblacion = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *padres = crear_poblacion(tamano_poblacion, longitud_genotipo);    
    poblacion *hijos = crear_poblacion(tamano_poblacion, longitud_genotipo);

    // Verify population initialization
    if (Poblacion == NULL || Poblacion->individuos == NULL) {
        fprintf(stderr, "Error: Population not properly initialized\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tamano_poblacion; i++) {
        if (Poblacion->individuos[i].genotipo == NULL) {
            fprintf(stderr, "Error: Genotype %d not properly initialized\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Crear estructuras equivalentes en GPU
    individuo_gpu *d_poblacion, *d_padres, *d_hijos;
    gpuErrchk(cudaMalloc(&d_poblacion, tamano_poblacion * sizeof(individuo_gpu)));
    gpuErrchk(cudaMalloc(&d_padres, tamano_poblacion * sizeof(individuo_gpu)));
    gpuErrchk(cudaMalloc(&d_hijos, tamano_poblacion * sizeof(individuo_gpu)));

    // Alocar memoria para los genotipos en GPU
    int **d_genotipos_poblacion, **d_genotipos_padres, **d_genotipos_hijos;
    if (cudaMalloc(&d_genotipos_poblacion, tamano_poblacion * sizeof(int*)) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory for genotype pointers\n");
    exit(EXIT_FAILURE);
    }

    // Alocar memoria para cada genotipo individual
    for(int i = 0; i < tamano_poblacion; i++) {
        // Para población
        int *d_genotipo_poblacion;
        gpuErrchk(cudaMalloc(&d_genotipo_poblacion, longitud_genotipo * sizeof(int)));
        gpuErrchk(cudaMemcpy(d_genotipo_poblacion, Poblacion->individuos[i].genotipo, 
                            longitud_genotipo * sizeof(int), 
                            cudaMemcpyHostToDevice));
        
        int *temp_ptr_poblacion = d_genotipo_poblacion;
        gpuErrchk(cudaMemcpy(d_genotipos_poblacion + i, &temp_ptr_poblacion, 
                            sizeof(int*), 
                            cudaMemcpyHostToDevice));

        // Para padres
        int *d_genotipo_padres;
        gpuErrchk(cudaMalloc(&d_genotipo_padres, longitud_genotipo * sizeof(int)));
        gpuErrchk(cudaMemcpy(d_genotipo_padres, padres->individuos[i].genotipo, 
                            longitud_genotipo * sizeof(int), 
                            cudaMemcpyHostToDevice));
        
        int *temp_ptr_padres = d_genotipo_padres;
        gpuErrchk(cudaMemcpy(d_genotipos_padres + i, &temp_ptr_padres, 
                            sizeof(int*), 
                            cudaMemcpyHostToDevice));

        // Para hijos
        int *d_genotipo_hijos;
        gpuErrchk(cudaMalloc(&d_genotipo_hijos, longitud_genotipo * sizeof(int)));
        gpuErrchk(cudaMemcpy(d_genotipo_hijos, hijos->individuos[i].genotipo, 
                            longitud_genotipo * sizeof(int), 
                            cudaMemcpyHostToDevice));
        
        int *temp_ptr_hijos = d_genotipo_hijos;
        gpuErrchk(cudaMemcpy(d_genotipos_hijos + i, &temp_ptr_hijos, 
                            sizeof(int*), 
                            cudaMemcpyHostToDevice));
    }

    // Verificar asignación de memoria
    for (int i = 0; i < tamano_poblacion; i++) {
        if (Poblacion->individuos[i].genotipo == NULL) {
            fprintf(stderr, "Error: genotipo del individuo %d no está asignado\n", i);
            exit(EXIT_FAILURE);
        }
    }

    // Creamos permutaciones aleatorias iniciales
    crear_permutaciones(Poblacion, longitud_genotipo);

    // Copiamos la población inicial a GPU
    gpuErrchk(cudaMemcpy(d_poblacion, Poblacion->individuos, 
              tamano_poblacion * sizeof(individuo_gpu), cudaMemcpyHostToDevice));

    // Evaluamos la población inicial
    evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_distancias, 
                                                      tamano_poblacion, longitud_genotipo);
    
    // Copiamos resultados de vuelta a CPU para ordenamiento
    gpuErrchk(cudaMemcpy(Poblacion->individuos, d_poblacion,
              tamano_poblacion * sizeof(individuo), cudaMemcpyDeviceToHost));

    // Ordenamos la población inicial
    ordenar_poblacion(Poblacion);

    // Inicializamos el mejor individuo
    individuo *Mejor_Individuo = (individuo *)malloc(sizeof(individuo));
    Mejor_Individuo->genotipo = (int *)malloc(longitud_genotipo * sizeof(int));

    // Copiamos el mejor individuo inicial
    for (int i = 0; i < longitud_genotipo; i++) {
        Mejor_Individuo->genotipo[i] = Poblacion->individuos[0].genotipo[i];
    }
    Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;
    
    // Ejecutamos el algoritmo genético
    for(int generacion = 0; generacion < num_generaciones; generacion++) {
        // Selección
        seleccionar_padres_kernel<<<gridSize, blockSize>>>(d_poblacion, d_padres, NULL,
                                                        num_competidores, tamano_poblacion,
                                                        longitud_genotipo, d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Cruzamiento
        cruzar_individuos_kernel<<<gridSize, blockSize>>>(d_padres, d_hijos, d_distancias,
                                                    probabilidad_cruce, tamano_poblacion,
                                                    longitud_genotipo, m, d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Mutación
        mutar_individuos_kernel<<<gridSize, blockSize>>>(d_hijos, d_distancias,
                                                    probabilidad_mutacion, tamano_poblacion,
                                                    longitud_genotipo, d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Copiar resultados a CPU
        gpuErrchk(cudaMemcpy(hijos->individuos, d_hijos,
                tamano_poblacion * sizeof(individuo), cudaMemcpyDeviceToHost));

        // Actualizar población
        actualizar_poblacion(&Poblacion, hijos, longitud_genotipo);

        // Evaluar nueva población en GPU
        gpuErrchk(cudaMemcpy(d_poblacion, Poblacion->individuos,
                tamano_poblacion * sizeof(individuo), cudaMemcpyHostToDevice));
        
        evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_distancias,
                                                    tamano_poblacion, longitud_genotipo);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Copiar resultados para ordenamiento
        gpuErrchk(cudaMemcpy(Poblacion->individuos, d_poblacion,
                tamano_poblacion * sizeof(individuo), cudaMemcpyDeviceToHost));

        ordenar_poblacion(Poblacion);

        // Actualizar mejor individuo
        if (Poblacion->individuos[0].fitness < Mejor_Individuo->fitness) {
            memcpy(Mejor_Individuo->genotipo, Poblacion->individuos[0].genotipo, 
                longitud_genotipo * sizeof(int));
            Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;
        }
    }

    // Imprimimos al mejor individuo
    printf("Mejor Individuo: ");
    for (int i = 0; i < longitud_genotipo; i++) {
        printf("%d  ", Mejor_Individuo->genotipo[i]);
    }
    printf("\nFitness: %f\n", Mejor_Individuo->fitness);

    // Imprimimos el recorrido
    printf("Recorrido: ");
    for (int i = 0; i < longitud_genotipo; i++) {
        printf("%s -> ", nombres_ciudades[Mejor_Individuo->genotipo[i]]);
    }
    printf("%s\n", nombres_ciudades[Mejor_Individuo->genotipo[0]]);

    // Liberamos memoria de CPU
    liberar_poblacion(Poblacion);
    liberar_poblacion(padres);
    liberar_poblacion(hijos);
    for (int i = 0; i < longitud_genotipo; i++) {
        free(distancias[i]);
    }
    free(distancias);
    free(h_distancias_flat);
    free(Mejor_Individuo->genotipo);
    free(Mejor_Individuo);

    // Liberamos memoria de GPU

    for(int i = 0; i < tamano_poblacion; i++) {
        int *d_genotipo;
        gpuErrchk(cudaMemcpy(&d_genotipo, &d_genotipos_poblacion[i], sizeof(int*), cudaMemcpyDeviceToHost));
        cudaFree(d_genotipo);
        gpuErrchk(cudaMemcpy(&d_genotipo, &d_genotipos_padres[i], sizeof(int*), cudaMemcpyDeviceToHost));
        cudaFree(d_genotipo);
        gpuErrchk(cudaMemcpy(&d_genotipo, &d_genotipos_hijos[i], sizeof(int*), cudaMemcpyDeviceToHost));
        cudaFree(d_genotipo);
    }

    cudaFree(d_genotipos_poblacion);
    cudaFree(d_genotipos_padres);
    cudaFree(d_genotipos_hijos);

    cudaFree(d_distancias);
    cudaFree(d_states);
    cudaFree(d_poblacion);
    cudaFree(d_padres);
    cudaFree(d_hijos);

    // Finalizamos la medición del tiempo
    time_t fin = time(NULL);
    double tiempo_ejecucion = difftime(fin, inicio);
    printf("Tiempo de ejecución: %.2f segundos\n", tiempo_ejecucion);

    return 0;
}