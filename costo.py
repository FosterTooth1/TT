import numpy as np

def costo(individuo, distancias):
    total_cost = 0.0
    for i in range(len(individuo) - 1):
        total_cost += distancias[individuo[i], individuo[i + 1]]
    total_cost += distancias[individuo[-1], individuo[0]]
    return total_cost

# Carga de datos
try:
    distancias = np.loadtxt('Distancias_no_head.csv', delimiter=',')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

individuo = [24,  1,  2, 23,  8,  16,  12,  7,  10,  18,  4,  25,  3,  21,  29,  28,  19,  27,  11,  31,  15,  13,  14,  20,  9,  22,  0,  30,  26,  17,  6,  5]
individuo = np.array(individuo)
costo_total = costo(individuo, distancias)
print(f"Costo total: {costo_total}")