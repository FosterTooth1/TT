import joblib
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import ctypes
from ctypes import c_int, c_double, c_char_p, POINTER, Structure, c_char, cast
import os
import matplotlib.pyplot as plt

class ResultadoRecocido(Structure):
    _fields_ = [("recorrido", POINTER(c_int)),
        ("fitness", c_double),
        ("tiempo_ejecucion", c_double),
        ("longitud_recorrido", c_int),
        ("fitness_generaciones", POINTER(c_double)),
        ("temperatura_inicial", c_double),
        ("temperatura_final", c_double),]

class AlgoritmoRecocido:
    def __init__(self, ruta_biblioteca):
        self.biblioteca = ctypes.CDLL(ruta_biblioteca)
        
        # Configuración de tipos igual que en Genético/PSO
        self.biblioteca.ejecutar_algoritmo_recocido.restype = POINTER(ResultadoRecocido)
        self.biblioteca.ejecutar_algoritmo_recocido.argtypes = [
            c_int,      # longitud_ruta
            c_int,      # num_generaciones
            c_double,   # tasa_enfriamiento
            c_double,   # temperatura_final
            c_int,      # max_neighbours
            c_int,      # m
            c_char_p,   # nombre_archivo
            c_int       # heuristica
        ]
        self.biblioteca.liberar_resultado.argtypes = [POINTER(ResultadoRecocido)]  # Mismo nombre de función

    def ejecutar(self, longitud_ruta, num_generaciones, tasa_enfriamiento,
               temperatura_final, max_neighbours, m, nombre_archivo, heuristica):
        try:
            nombre_archivo_bytes = nombre_archivo.encode('utf-8')
            
            resultado_ptr = self.biblioteca.ejecutar_algoritmo_recocido(
                c_int(longitud_ruta),
                c_int(num_generaciones),
                c_double(tasa_enfriamiento),
                c_double(temperatura_final),
                c_int(max_neighbours),
                c_int(m),
                nombre_archivo_bytes,
                c_int(heuristica)
            )
            
            if not resultado_ptr:
                raise RuntimeError("Error en ejecución del Recocido")
            
            resultado = resultado_ptr.contents
            
            # Copia de datos
            recorrido = [resultado.recorrido[i] for i in range(resultado.longitud_recorrido)]
            
            fitness_hist = [resultado.fitness_generaciones[i] for i in range(num_generaciones)]
            
            salida = {
                'recorrido': recorrido,
                'fitness': resultado.fitness,
                'tiempo_ejecucion': resultado.tiempo_ejecucion,
                'fitness_generaciones': fitness_hist,
                'temperatura_inicial': resultado.temperatura_inicial,
                'temperatura_final': resultado.temperatura_final
            }
            
            self.biblioteca.liberar_resultado(resultado_ptr)
            
            return salida
            
        except Exception as e:
            raise RuntimeError(f"Error en Recocido Simulado: {str(e)}")

def cargar_CSV(nombre_archivo):
    try:
        df = pd.read_csv(nombre_archivo)
        return df
    except FileNotFoundError:
        print(f"El archivo {nombre_archivo} no se encuentra.")
        return None
    
def cambiarFormatoHora(fecha_hora_str):
    dt = datetime.strptime(fecha_hora_str, "%Y-%m-%d %H:%M")
    return dt.hour * 60 + dt.minute

def cambiarFormatoViento(dir):
    mapa = {"CALM": -1, "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90,
            "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5,
            "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}
    return mapa.get(dir)

def realizar_predicciones(modelo, df_naves_industriales_filtrado, api_key):
    condiciones = ['Nublado', 'Considerablemente nublado', 'Despejado', 'Niebla', 'Bruma',
        'Lluvia intensa', 'Tormenta electrica intensa', 'Neblina', 'Lluvia', 'Truenos']
    predicciones = []
    # Recorremos con iterrows sobre copia para no modificar original
    for idx, fila in df_naves_industriales_filtrado.iterrows():
        lat = fila.get('latitud')
        lon = fila.get('longitud')
        # Validar lat/lon: no nulos, rango correcto, y evitar (0,0) si es un dato inválido
        if pd.isnull(lat) or pd.isnull(lon):
            print(f"[Aviso] Fila índice {idx}: latitud/longitud faltante, se omite predicción.")
            predicciones.append(None)
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            print(f"[Aviso] Fila índice {idx}: lat/long no convertible a float: ({lat}, {lon}).")
            predicciones.append(None)
            continue
        # Rango válido
        if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
            print(f"[Aviso] Fila índice {idx}: lat/lon fuera de rango: ({lat_f}, {lon_f}).")
            predicciones.append(None)
            continue
        # Evitar (0,0) como coordenadas inválidas
        if lat_f == 0.0 and lon_f == 0.0:
            print(f"[Aviso] Fila índice {idx}: coordenadas (0.0,0.0) se consideran inválidas, se omite.")
            predicciones.append(None)
            continue

        location = f"{lat_f}, {lon_f}"
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
        try:
            response = requests.get(url, timeout=10)
        except Exception as e:
            print(f"[Error] Fila índice {idx}: excepción en requests.get para {location}: {e}")
            predicciones.append(None)
            continue

        if response.status_code != 200:
            print(f"[Error] Fila índice {idx}: status_code={response.status_code} en la petición para {location}.")
            predicciones.append(None)
            continue

        try:
            data = response.json()
        except ValueError:
            print(f"[Error] Fila índice {idx}: no se pudo parsear JSON para {location}.")
            predicciones.append(None)
            continue

        # Extraemos los campos necesarios, manejando posibles faltantes
        try:
            hora_API = data['location']['localtime']  # "YYYY-MM-DD HH:MM"
            temperatura = data['current']['temp_c']
            puntoRocio = data['current']['dewpoint_c']
            humedad = data['current']['humidity']
            dirViento_API = data['current']['wind_dir']
            velocidadViento = data['current']['wind_kph']
        except KeyError as e:
            print(f"[Error] Fila índice {idx}: faltante clave en JSON: {e}")
            predicciones.append(None)
            continue
        # Transformar hora y viento:
        try:
            hora_num = cambiarFormatoHora(hora_API)
        except Exception as e:
            print(f"[Error] Fila índice {idx}: cambiarFormatoHora fallo con '{hora_API}': {e}")
            predicciones.append(None)
            continue

        dir_viento_num = cambiarFormatoViento(dirViento_API)
        if dir_viento_num is None:
            # Si no se reconoce la dirección, mostrar aviso
            print(f"[Aviso] Fila índice {idx}: dirección de viento '{dirViento_API}' no reconocida.")
        # Construir DataFrame de un solo registro
        columnas = ['Time', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed']
        fila_predict = pd.DataFrame([[
            hora_num,
            temperatura,
            puntoRocio,
            humedad,
            dir_viento_num,
            velocidadViento
        ]], columns=columnas)

        # Llamar a predict
        try:
            pred_array = modelo.predict(fila_predict)
            if len(pred_array) == 0:
                print(f"[Error] Fila índice {idx}: modelo.predict devolvió array vacío.")
                predicciones.append(None)
                continue
            pred_idx = pred_array[0]
        except Exception as e:
            print(f"[Error] Fila índice {idx}: excepción en modelo.predict: {e}")
            predicciones.append(None)
            continue

        # Mapear a texto si es índice válido
        if isinstance(pred_idx, (int, np.integer)) and 0 <= pred_idx < len(condiciones):
            etiqueta = condiciones[pred_idx]
        else:
            print(f"[Aviso] Fila índice {idx}: índice predicción {pred_idx} fuera de rango o no entero.")
            etiqueta = str(pred_idx)
        predicciones.append(etiqueta)

    # Asignar columna en copia
    df = df_naves_industriales_filtrado.copy()
    df.loc[:, 'Prediccion'] = predicciones
    return df

def decimal_to_hhmm(val):
    horas = int(val)
    minutos = int(round((val - horas) * 60))
    # Ajuste si el redondeo llega a 60
    if minutos >= 60:
        horas += minutos // 60
        minutos = minutos % 60
    return f"{horas:02d}:{minutos:02d}"
        
def main(): # Rutas a los archivos de data
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_csv_naves = os.path.join(directorio_actual, "Naves_Industriales.csv")
    ruta_csv_predicciones = os.path.join(directorio_actual, "Naves_Industriales_Con_Predicciones.csv")
    ruta_matriz_distancias = os.path.join(directorio_actual, "Matriz_Distancias_Carretera.csv")
    ruta_modelo = os.path.join(directorio_actual, "prediccion_clima.pkl")

    # Cargar el CSV con la informacion completa de las naves industriales
    df_naves_industriales = cargar_CSV(ruta_csv_naves)
    
    print("Listado de Naves Industriales:")
    print(df_naves_industriales)
    
    # Ingresar el listado de números con ","; si se ingresa -1 se seleccionan todas
    entrada_usuario = input("Selecciona los índices de las naves industriales, separados por comas (ej. 0,1,3,5,7), si ingresa -1 se seleccionan todos: ")    
    
    indices_seleccionados = [int(indice.strip()) for indice in entrada_usuario.split(',')]
    
    if indices_seleccionados == [-1]:
        indices_seleccionados = list(range(len(df_naves_industriales)))

    df_naves_industriales_filtrado = df_naves_industriales.iloc[indices_seleccionados]

    # Leer directamente la matriz de distancias
    df_matriz_distancias = pd.read_csv(ruta_matriz_distancias)
    
    # Cargar modelo para predicciones climáticas
    Modelo_RandomForest = joblib.load(ruta_modelo)

    api_key = "7f25124e580c4de6a2e00312251205"
    
    realizar_predicciones_climaticas = input("¿Desea realizar predicciones climáticas para las naves industriales seleccionadas? (s/n): ").strip().lower()
    if realizar_predicciones_climaticas == 's':
        inicio_tiempo = datetime.now()
        print("Realizando predicciones climáticas para las naves industriales seleccionadas...")

        df_naves_industriales_filtrado = realizar_predicciones(Modelo_RandomForest, df_naves_industriales_filtrado, api_key)

        fin_tiempo = datetime.now()
        duracion = fin_tiempo - inicio_tiempo
        print(f"Predicciones completadas en {duracion.total_seconds():.2f} segundos.")

        df_naves_industriales_filtrado.to_csv(ruta_csv_predicciones, index=False)
        print(f"Se han guardado las predicciones climáticas en '{ruta_csv_predicciones}'.") 
    else:
        df_naves_industriales_filtrado = cargar_CSV(ruta_csv_predicciones)
        print(f"Se han cargado las predicciones climáticas desde '{ruta_csv_predicciones}'.")

    # Penalizaciones por condiciones climáticas
    penalizaciones = {'Nublado': 1.1,
                      'Considerablemente nublado': 1.2,
                      'Despejado': 1.0,
                      'Niebla': 1.8,
                      'Bruma': 1.3,
                      'Lluvia intensa': 1.7,
                      'Tormenta electrica intensa': 1.9,
                      'Neblina': 1.6,
                      'Lluvia': 1.5,
                      'Truenos': 1.4}
    
    lambda_penalizacion = 1.0
    
    aplicar_penalizaciones = input("¿Desea aplicar penalizaciones por condiciones climáticas? (s/n): ").strip().lower()
    
    if aplicar_penalizaciones == 's':
        for pos, fila in df_naves_industriales_filtrado.reset_index(drop=True).iterrows():
            pred = fila['Prediccion']
            if pred in penalizaciones:
                penal = penalizaciones[pred] * lambda_penalizacion
                # Multiplicar fila y columna usando .iloc
                df_matriz_distancias.iloc[pos, :] *= penal   # fila
                df_matriz_distancias.iloc[:, pos] *= penal   # columna
                df_matriz_distancias.iloc[pos, pos] = 0.0   # diagonal
    else:   
        print("No se aplicarán penalizaciones por condiciones climáticas.")

    print("Naves Industriales disponibles para iniciar el recorrido:")
    print(df_naves_industriales_filtrado)
       
    num_naves = len(df_naves_industriales_filtrado)
    print(f"Número total de naves industriales seleccionadas: {num_naves}")
    
    nombre_biblioteca = "recocido.dll" if os.name == 'nt' else "librecocido.so"
    ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
    
    print(f"Llamando a DLL con ruta: {ruta_biblioteca}")

    rs = AlgoritmoRecocido(ruta_biblioteca)
    
    params = {'longitud_ruta': num_naves,
              'num_generaciones': 800,
              'tasa_enfriamiento': 0.99,
              'temperatura_final': 0.001,
              'max_neighbours': num_naves * 10,
              'm': 3,
              'nombre_archivo': "Matriz_Distancias_Carretera.csv",
              'heuristica': 0}
        
    resultado = rs.ejecutar(**params)
    
    print("\nRecorrido óptimo encontrado (índices):")
    print(resultado['recorrido'])
    print("\nRecorrido óptimo encontrado (nombres de naves industriales):")
    for idx in resultado['recorrido']:
        nombre_nave = df_naves_industriales_filtrado.iloc[idx]['nombre']
        print(f"- {nombre_nave}")
    print(f"\nFitness: {resultado['fitness']:.2f}")
    print(f"Tiempo: {resultado['tiempo_ejecucion']:.2f}s")
    print(f"Temperatura inicial: {resultado['temperatura_inicial']:.2f}")
    print(f"Temperatura final: {resultado['temperatura_final']:.5f}")
    
    # Mostrar tabla del fitness durante las generaciones
    plt.plot(resultado['fitness_generaciones'])
    plt.title("Evolución del Fitness - Recocido Simulado")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.show()
    

    salida_json = []
    for idx in resultado['recorrido']:
        fila = df_naves_industriales_filtrado.iloc[idx]
        salida_json.append({
            "lat": float(fila['latitud']),
            "lng": float(fila['longitud']),
            "nombre": fila['nombre'],
            "condicion": fila['Prediccion']
        })

    with open("ruta_Ejemplo.json", "w", encoding="utf-8") as f:
        json.dump(salida_json, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()


