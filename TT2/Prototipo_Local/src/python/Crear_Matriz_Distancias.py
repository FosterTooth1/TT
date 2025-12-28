import requests
import pandas as pd
import time
import numpy as np
import math
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde la raíz del proyecto
directorio_script = os.path.dirname(os.path.abspath(__file__))
directorio_proyecto = os.path.dirname(os.path.dirname(directorio_script))
ruta_env = os.path.join(directorio_proyecto, ".env")
load_dotenv(ruta_env)
# API Key de OpenRouteService
OPENROUTESERVICE_API_KEY = os.getenv("OPENROUTESERVICE_API_KEY")

# Tamaño de los lotes para las peticiones
chunk_size = 50

# Cargar el CSV de las naves industriales 
try:
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    directorio_base = os.path.dirname(os.path.dirname(directorio_actual))
    ruta_csv = os.path.join(directorio_base, 'data', 'processed', 'Naves_Industriales.csv')
    df = pd.read_csv(ruta_csv)
except FileNotFoundError:
    print("Error: No se encontró el archivo 'Naves_Industriales.csv'.")
    exit()

# Preparar los datos
coordinates = df[['longitud', 'latitud']].values.tolist()
location_names = df['nombre'].tolist()
num_locations = len(coordinates)

# Crear una matriz que almacenará las distancias finales
final_matrix = np.zeros((num_locations, num_locations))

# Configurar la URL y cabeceras de la API
headers = {
    'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
    'Authorization': OPENROUTESERVICE_API_KEY,
    'Content-Type': 'application/json; charset=utf-8'
}
url = 'https://api.openrouteservice.org/v2/matrix/driving-car'

total_chunks = math.ceil(num_locations / chunk_size)
total_requests = total_chunks * total_chunks
request_count = 0

print(f"Iniciando el proceso por lotes. Se realizarán {total_requests} peticiones a la API.")

try:
    # Iterar sobre los lotes de origen (filas de la matriz)
    for i in range(0, num_locations, chunk_size):
        # Iterar sobre los lotes de destino (columnas de la matriz)
        for j in range(0, num_locations, chunk_size):
            request_count += 1
            print(f"Procesando petición {request_count}/{total_requests}...")

            # Definir los índices de origen y destino para este lote
            origin_indices = list(range(i, min(i + chunk_size, num_locations)))
            dest_indices = list(range(j, min(j + chunk_size, num_locations)))
            
            # Combinar coordenadas de origen y destino para la petición
            locations_chunk = [coordinates[k] for k in origin_indices] + [coordinates[k] for k in dest_indices]
            sources_chunk = list(range(len(origin_indices)))
            destinations_chunk = list(range(len(origin_indices), len(locations_chunk)))

            # Crear el cuerpo de la petición
            json_body = {
                'locations': locations_chunk,
                'sources': sources_chunk,
                'destinations': destinations_chunk,
                'metrics': ['distance'],
                'units': 'km'
            }

            # Realizar la petición
            response = requests.post(url, json=json_body, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            distance_chunk = data['distances']
            
            # Rellenar la matriz final con los datos obtenidos
            for row_idx, origin_master_idx in enumerate(origin_indices):
                for col_idx, dest_master_idx in enumerate(dest_indices):
                    final_matrix[origin_master_idx, dest_master_idx] = distance_chunk[row_idx][col_idx]

            # Pausa para no exceder el límite de peticiones por minuto
            time.sleep(10.5)

    # Crear el DataFrame final con los resultados completos
    dist_matrix_df = pd.DataFrame(final_matrix, index=location_names, columns=location_names)
    
    print("\nProceso completado exitosamente")
    print(dist_matrix_df)

    # Guardar el resultado
    output_filename = os.path.join(directorio_base, "data", "processed", "Matriz_Distancias_Carretera.csv")
    dist_matrix_df.to_csv(output_filename, header=False, index=False)
    print(f"\nMatriz guardada exitosamente en el archivo: '{output_filename}'")

except requests.exceptions.HTTPError as http_err:
    print(f"\nError HTTP: {http_err}")
    print(f"Respuesta del servidor: {response.text}")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")