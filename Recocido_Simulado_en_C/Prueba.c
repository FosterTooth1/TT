#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

// --- Estructuras ---
typedef struct {
    int *ruta;
    double fitness;
} Solucion;

// --- Prototipos ---
Solucion *crear_solucion(int longitud);
void liberar_solucion(Solucion *sol);
double calcular_fitness(int *ruta, double **distancias, int n);
void copiar_ruta(int *dest, int *src, int n);

// Vecindarios
void vecino_2opt(int *ruta_act, int *ruta_vec, int n);
void vecino_3opt(int *ruta_act, int *ruta_vec, int n);
void vecino_double_bridge(int *ruta_act, int *ruta_vec, int n);
void generar_vecino(int *ruta_act, int *ruta_vec, int n, double temperatura);

// Enfriamiento adaptativo + reheating
double enfriamiento(double temp, int iter);

// Iterated Local Search / Multi-start
Solucion *simulated_annealing(double **distancias, int n,
                               double temp_init, double temp_min,
                               int max_iter, int reheating_patience,
                               double reheating_factor);

int main() {
    srand(time(NULL));
    int n = 32;
    // Parámetros SA
    double temp_init = n * n;
    double temp_min  = 1e-8;
    int max_iter = 30000;
    int reheating_patience = 1000;      // iter sin mejora antes de recalentar
    double reheating_factor  = 1.5;    // factor de recalentamiento
    int restarts = 5;                  // multi-start

    // Cargar distancias
    double **distancias = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) distancias[i] = malloc(n * sizeof(double));
    FILE *f = fopen("Distancias_no_head.csv","r");
    if (!f) { perror("Error abrir archivo"); return 1; }
    char buf[8192]; int fila=0;
    while(fgets(buf,sizeof(buf),f) && fila<n) {
        char *tok = strtok(buf, ","); int col=0;
        while(tok && col<n) {
            distancias[fila][col] = atof(tok);
            tok = strtok(NULL, ","); col++;
        }
        fila++;
    }
    fclose(f);

    // Multi-start: ejecutar varias SA y quedarnos con la mejor global
    Solucion *global_best = NULL;
    for (int r = 0; r < restarts; r++) {
        Solucion *sol = simulated_annealing(distancias, n,
                                            temp_init, temp_min,
                                            max_iter, reheating_patience,
                                            reheating_factor);
        if (!global_best || sol->fitness < global_best->fitness) {
            if (global_best) liberar_solucion(global_best);
            global_best = sol;
        } else {
            liberar_solucion(sol);
        }
    }

    // Mostrar mejor resultado
    printf("Mejor fitness global: %.2f\n", global_best->fitness);
    // Aquí puedes imprimir la ruta guardada en global_best->ruta...

    // Liberar
    for (int i=0;i<n;i++) free(distancias[i]);
    free(distancias);
    liberar_solucion(global_best);
    return 0;
}

// --- Implementaciones ---
Solucion *crear_solucion(int longitud) {
    Solucion *s = malloc(sizeof(Solucion));
    s->ruta = malloc(longitud * sizeof(int));
    s->fitness = 0.0;
    return s;
}

void liberar_solucion(Solucion *sol) {
    free(sol->ruta);
    free(sol);
}

void copiar_ruta(int *dest, int *src, int n) {
    memcpy(dest, src, n * sizeof(int));
}

double calcular_fitness(int *ruta, double **distancias, int n) {
    double total = 0.0;
    for (int i = 0; i < n-1; i++) total += distancias[ruta[i]][ruta[i+1]];
    total += distancias[ruta[n-1]][ruta[0]];
    return total;
}

// --- Vecindarios ---
void vecino_2opt(int *ruta_act, int *ruta_vec, int n) {
    copiar_ruta(ruta_vec, ruta_act, n);
    int i = rand() % (n-1);
    int j = rand() % (n - i) + i;
    while (i < j) { int t = ruta_vec[i]; ruta_vec[i] = ruta_vec[j]; ruta_vec[j] = t; i++; j--; }
}

void vecino_3opt(int *ruta_act, int *ruta_vec, int n) {
    copiar_ruta(ruta_vec, ruta_act, n);
    // Elegir tres puntos i<j<k
    int i = rand() % (n-2);
    int j = (rand() % (n-i-1)) + i + 1;
    int k = (rand() % (n-j-1)) + j + 1;
    // Tres segmentos [0,i), [i,j), [j,k), [k,n)
    // Reconectar de una de las 7 posibles formas (aquí simple: invertir cada segmento con prob)
    if (rand()%2) { // invertir segmento medio
        int a=i, b=j-1;
        while(a<b){int t=ruta_vec[a]; ruta_vec[a]=ruta_vec[b]; ruta_vec[b]=t; a++; b--;}
    }
    if (rand()%2) { // invertir último
        int a=j, b=k-1;
        while(a<b){int t=ruta_vec[a]; ruta_vec[a]=ruta_vec[b]; ruta_vec[b]=t; a++; b--;}
    }
}

void vecino_double_bridge(int *ruta_act, int *ruta_vec, int n) {
    copiar_ruta(ruta_vec, ruta_act, n);
    int pos[4];
    // elegir 4 puntos de separación equidistantes al azar
    pos[0] = rand() % (n/4);
    pos[1] = pos[0] + 1 + rand() % (n/4);
    pos[2] = pos[1] + 1 + rand() % (n/4);
    pos[3] = pos[2] + 1 + rand() % (n/4);
    int tmp[n]; int idx=0;
    // reconectar: 0-pos0, pos3-pos2, pos1-pos0, pos2-pos1 en nuevo orden
    memcpy(tmp + idx, ruta_vec, pos[0]*sizeof(int)); idx += pos[0];
    memcpy(tmp + idx, ruta_vec+pos[2], (pos[3]-pos[2])*sizeof(int)); idx += pos[3]-pos[2];
    memcpy(tmp + idx, ruta_vec+pos[1], (pos[2]-pos[1])*sizeof(int)); idx += pos[2]-pos[1];
    memcpy(tmp + idx, ruta_vec+pos[0], (pos[1]-pos[0])*sizeof(int)); idx += pos[1]-pos[0];
    memcpy(tmp + idx, ruta_vec+pos[3], (n-pos[3])*sizeof(int));
    memcpy(ruta_vec, tmp, n*sizeof(int));
}

void generar_vecino(int *ruta_act, int *ruta_vec, int n, double temperatura) {
    // Seleccionar operador:
    double r = (double)rand() / RAND_MAX;
    if (r < 0.6) vecino_2opt(ruta_act, ruta_vec, n);
    else if (r < 0.9) vecino_3opt(ruta_act, ruta_vec, n);
    else vecino_double_bridge(ruta_act, ruta_vec, n);
}

// --- Enfriamiento adaptativo + reheating ---
double enfriamiento(double temp, int iter) {
    // Cauchy cooling
    return temp / (1.0 + iter);
}

// --- Simulated Annealing con reheating y retorno ---
Solucion *simulated_annealing(double **dist, int n,
                               double temp_init, double temp_min,
                               int max_iter, int patience,
                               double heat_factor) {
    Solucion *actual = crear_solucion(n);
    Solucion *mejor  = crear_solucion(n);
    // solución inicial aleatoria
    for (int i=0;i<n;i++) actual->ruta[i]=i;
    // fisher-yates
    for (int i=n-1;i>0;i--) { int j=rand()%(i+1); int t=actual->ruta[i]; actual->ruta[i]=actual->ruta[j]; actual->ruta[j]=t; }
    actual->fitness = calcular_fitness(actual->ruta, dist, n);
    copiar_ruta(mejor->ruta, actual->ruta, n);
    mejor->fitness = actual->fitness;
    
    double T = temp_init;
    int last_improve = 0;
    int *vec = malloc(n*sizeof(int));

    for (int iter=1; iter<=max_iter && T>temp_min; iter++) {
        generar_vecino(actual->ruta, vec, n, T);
        double f_vec = calcular_fitness(vec, dist, n);
        double delta = f_vec - actual->fitness;
        double prob = (delta<0)?1.0:exp(-delta/T);
        if (((double)rand()/RAND_MAX) < prob) {
            copiar_ruta(actual->ruta, vec, n);
            actual->fitness = f_vec;
            if (f_vec < mejor->fitness) {
                copiar_ruta(mejor->ruta, actual->ruta, n);
                mejor->fitness = f_vec;
                last_improve = iter;
            }
        }
        // reheating si sin mejora
        if (iter - last_improve > patience) {
            T *= heat_factor;
            last_improve = iter;
        }
        // enfriamiento Cauchy
        T = enfriamiento(T, iter);
    }

    free(vec);
    liberar_solucion(actual);
    return mejor;
}
