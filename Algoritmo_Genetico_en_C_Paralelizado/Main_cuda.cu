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

    // Verificar memoria GPU disponible
    size_t memoria_libre, memoria_total;
    cudaMemGetInfo(&memoria_libre, &memoria_total);
    printf("Memoria GPU disponible: %zu bytes\n", memoria_libre);

    // Calcular memoria requerida
    size_t memoria_requerida = tamano_poblacion * longitud_genotipo * sizeof(int);
    if (memoria_requerida > memoria_libre) {
        fprintf(stderr, "Error: No hay suficiente memoria GPU\n");
        exit(1);
    }

    // Nombre del archivo con las distancias
    const char *nombre_archivo = "Distancias_no_head.csv";

    // Reservamos memoria para la matriz de distancias
    double **distancias = (double **)malloc(longitud_genotipo * sizeof(double *));
    for (int i = 0; i < longitud_genotipo; i++) {
        distancias[i] = (double *)malloc(longitud_genotipo * sizeof(double));
    }

    // Leer archivo de distancias
    FILE *archivo = fopen(nombre_archivo, "r");
    if (!archivo) {
        perror("Error al abrir el archivo");
        return 1;
    }

    // Leer y llenar la matriz de distancias
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

    // Nombres de las ciudades
    const char nombres_ciudades[32][20] = {
        "Aguascalientes", "Baja California", "Baja California Sur",
        "Campeche", "Chiapas", "Chihuahua", "Coahuila", "Colima", "Durango",
        "Guanajuato", "Guerrero", "Hidalgo", "Jalisco", "Estado de México",
        "Michoacán", "Morelos", "Nayarit", "Nuevo León", "Oaxaca", "Puebla",
        "Querétaro", "Quintana Roo", "San Luis Potosí", "Sinaloa", "Sonora",
        "Tabasco", "Tamaulipas", "Tlaxcala", "Veracruz", "Yucatán",
        "Zacatecas", "CDMX"
    };

    // Aplanar matriz de distancias para GPU
    double *h_distancias_flat = (double*)malloc(longitud_genotipo * longitud_genotipo * sizeof(double));
    for(int i = 0; i < longitud_genotipo; i++) {
        for(int j = 0; j < longitud_genotipo; j++) {
            h_distancias_flat[i * longitud_genotipo + j] = distancias[i][j];
        }
    }

    // Asignación de memoria en GPU
    double *d_distancias;
    curandState *d_states;
    int *d_genotipos;
    
    // Asignar memoria para distancias y estados random
    gpuErrchk(cudaMalloc(&d_distancias, longitud_genotipo * longitud_genotipo * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_states, tamano_poblacion * sizeof(curandState)));
    gpuErrchk(cudaMalloc(&d_genotipos, tamano_poblacion * longitud_genotipo * sizeof(int)));

    // Copiar distancias a GPU
    gpuErrchk(cudaMemcpy(d_distancias, h_distancias_flat, 
              longitud_genotipo * longitud_genotipo * sizeof(double), 
              cudaMemcpyHostToDevice));

    // Configuración de bloques CUDA
    int blockSize, minGridSize, gridSize;
    obtenerConfiguracionCUDA(&blockSize, &minGridSize, &gridSize, tamano_poblacion);

    // Inicializar estados random en GPU
    setup_curand_kernel<<<gridSize, blockSize>>>(d_states, time(NULL));
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    // Crear poblaciones en CPU
    poblacion *Poblacion = crear_poblacion(tamano_poblacion, longitud_genotipo);
    poblacion *padres = crear_poblacion(tamano_poblacion, longitud_genotipo);    
    poblacion *hijos = crear_poblacion(tamano_poblacion, longitud_genotipo);

    // Verificar inicialización de población
    if (!Poblacion || !Poblacion->individuos) {
        fprintf(stderr, "Error: Población no inicializada correctamente\n");
        exit(EXIT_FAILURE);
    }

    // Crear estructuras en GPU
    individuo_gpu *d_poblacion, *d_padres, *d_hijos;
    gpuErrchk(cudaMalloc(&d_poblacion, tamano_poblacion * sizeof(individuo_gpu)));
    gpuErrchk(cudaMalloc(&d_padres, tamano_poblacion * sizeof(individuo_gpu)));
    gpuErrchk(cudaMalloc(&d_hijos, tamano_poblacion * sizeof(individuo_gpu)));

    // Crear permutaciones aleatorias iniciales
    crear_permutaciones(Poblacion, longitud_genotipo);

    // Copiar población inicial a GPU
    gpuErrchk(cudaMemcpy(d_poblacion, Poblacion->individuos, 
              tamano_poblacion * sizeof(individuo_gpu), cudaMemcpyHostToDevice));

    // Evaluar población inicial
    evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_genotipos, 
                                                     d_distancias, tamano_poblacion, 
                                                     longitud_genotipo);
    
    // Copiar resultados de vuelta a CPU
    gpuErrchk(cudaMemcpy(Poblacion->individuos, d_poblacion,
              tamano_poblacion * sizeof(individuo), cudaMemcpyDeviceToHost));

    // Ordenar población inicial
    ordenar_poblacion(Poblacion);

    // Inicializar mejor individuo
    individuo *Mejor_Individuo = (individuo *)malloc(sizeof(individuo));
    Mejor_Individuo->genotipo = (int *)malloc(longitud_genotipo * sizeof(int));
    memcpy(Mejor_Individuo->genotipo, Poblacion->individuos[0].genotipo, 
           longitud_genotipo * sizeof(int));
    Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;

    // Bucle principal del algoritmo genético
    for(int generacion = 0; generacion < num_generaciones; generacion++) {
        // Selección de padres
        seleccionar_padres_kernel<<<gridSize, blockSize>>>(d_poblacion, d_padres, 
                                                          d_genotipos, num_competidores, 
                                                          tamano_poblacion, longitud_genotipo, 
                                                          d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Cruzamiento
        cruzar_individuos_kernel<<<gridSize, blockSize>>>(d_padres, d_hijos, 
                                                         d_distancias, probabilidad_cruce, 
                                                         tamano_poblacion, longitud_genotipo, 
                                                         m, d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Mutación
        mutar_individuos_kernel<<<gridSize, blockSize>>>(d_hijos, d_distancias,
                                                        probabilidad_mutacion, 
                                                        tamano_poblacion, longitud_genotipo, 
                                                        d_states);
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());

        // Actualizar población
        gpuErrchk(cudaMemcpy(hijos->individuos, d_hijos,
                  tamano_poblacion * sizeof(individuo), cudaMemcpyDeviceToHost));
        actualizar_poblacion(&Poblacion, hijos, longitud_genotipo);
        gpuErrchk(cudaMemcpy(d_poblacion, Poblacion->individuos,
                  tamano_poblacion * sizeof(individuo), cudaMemcpyHostToDevice));

        // Evaluar nueva población
        evaluar_poblacion_kernel<<<gridSize, blockSize>>>(d_poblacion, d_genotipos,
                                                         d_distancias, tamano_poblacion, 
                                                         longitud_genotipo);
        gpuErrchk(cudaMemcpy(Poblacion->individuos, d_poblacion,
                  tamano_poblacion * sizeof(individuo), cudaMemcpyDeviceToHost));

        ordenar_poblacion(Poblacion);

        // Actualizar mejor individuo si se encuentra uno mejor
        if (Poblacion->individuos[0].fitness < Mejor_Individuo->fitness) {
            memcpy(Mejor_Individuo->genotipo, Poblacion->individuos[0].genotipo, 
                   longitud_genotipo * sizeof(int));
            Mejor_Individuo->fitness = Poblacion->individuos[0].fitness;
        }
    }

    // Imprimir resultados
    printf("\nMejor recorrido encontrado:\n");
    for (int i = 0; i < longitud_genotipo; i++) {
        printf("%s -> ", nombres_ciudades[Mejor_Individuo->genotipo[i]]);
    }
    printf("%s\n", nombres_ciudades[Mejor_Individuo->genotipo[0]]);
    printf("Distancia total: %f\n", Mejor_Individuo->fitness);

    // Liberar memoria CPU
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

    // Liberar memoria GPU
    cudaFree(d_distancias);
    cudaFree(d_states);
    cudaFree(d_genotipos);
    cudaFree(d_poblacion);
    cudaFree(d_padres);
    cudaFree(d_hijos);

    // Imprimir tiempo de ejecución
    time_t fin = time(NULL);
    double tiempo_ejecucion = difftime(fin, inicio);
    printf("Tiempo de ejecución: %.2f segundos\n", tiempo_ejecucion);

    return 0;
}
