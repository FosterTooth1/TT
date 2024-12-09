import pandas as pd

# Cargar los datos del archivo Excel
file_path = "Resultados_Train_TT.xlsx"  # Cambia a la ruta donde esté tu archivo
data = pd.read_excel(file_path)

# Identificar las mejores combinaciones según los criterios
# 1. Menor tiempo
mejor_tiempo = data.sort_values(by='Tiempo', ascending=True).head(5)

# 2. Menor valor de "Mejor" (ahora buscamos minimizar esta métrica)
mejor_valor = data.sort_values(by='Mejor', ascending=True).head(5)

# 3. Menor desviación estándar
menor_desviacion = data.sort_values(by='Desviacion', ascending=True).head(5)

# Combinación ponderada: tiempo, "Mejor", y desviación
# Crear una métrica combinada (normalización para ponderación)
data['Metrica_Compuesta'] = (
    -data['Mejor'] / data['Mejor'].max() -  # Invertido porque queremos minimizar
    data['Tiempo'] / data['Tiempo'].max() - 
    data['Desviacion'] / data['Desviacion'].max()
)

# Seleccionar las mejores combinaciones según esta métrica
mejor_combinada = data.sort_values(by='Metrica_Compuesta', ascending=False).head(5)

# Mostrar resultados
print("Menor Tiempo:")
print(mejor_tiempo)

print("\nMenor Valor de 'Mejor':")
print(mejor_valor)

print("\nMenor Desviación:")
print(menor_desviacion)

print("\nMétrica Combinada:")
print(mejor_combinada)