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
from flask import Flask, request, jsonify, render_template

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
      
# Configuración de Flask
app = Flask(__name__)

# Variables globales para almacenar datos
df_naves_industriales = None
df_matriz_distancias_original = None
Modelo_RandomForest = None
api_key = "7f25124e580c4de6a2e00312251205"

def inicializar_datos():
    """Inicializa los datos necesarios para el sistema"""
    global df_naves_industriales, df_matriz_distancias_original, Modelo_RandomForest
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_csv_naves = os.path.join(directorio_actual, "Naves_Industriales.csv")
    ruta_matriz_distancias = os.path.join(directorio_actual, "Matriz_Distancias_Carretera.csv")
    ruta_modelo = os.path.join(directorio_actual, "prediccion_clima.pkl")
    
    # Cargar datos solo una vez
    if df_naves_industriales is None:
        df_naves_industriales = cargar_CSV(ruta_csv_naves)
        df_matriz_distancias_original = pd.read_csv(ruta_matriz_distancias, header=None)
        Modelo_RandomForest = joblib.load(ruta_modelo)

@app.route('/')
def index():
    """Página principal - redirige a seleccionar naves"""
    return render_template('SeleccionarNaves.html')

@app.route('/mejor-ruta')
def mejor_ruta():
    """Página de mejor ruta"""
    return render_template('MejorRuta.html')

@app.route('/iniciar-sesion')
def iniciar_sesion():
    """Página de iniciar sesión"""
    return render_template('IniciarSesion.html')

@app.route('/nueva-cuenta')
def nueva_cuenta():
    """Página de nueva cuenta"""
    return render_template('NuevaCuenta.html')

@app.route('/rutas-recientes')
def rutas_recientes():
    """Página de rutas recientes"""
    return render_template('RutasRecientes.html')

@app.route('/api/naves', methods=['GET'])
def obtener_naves():
    """API para obtener la lista de naves industriales"""
    inicializar_datos()
    if df_naves_industriales is None:
        return jsonify({"error": "No se pudieron cargar las naves industriales"}), 500
    
    naves = df_naves_industriales[['nombre', 'latitud', 'longitud']].to_dict('records')
    return jsonify(naves)

@app.route('/api/generar-ruta', methods=['POST'])
def generar_ruta():
    """API para generar la ruta optimizada basada en las naves seleccionadas"""
    try:
        data = request.get_json()
        indices_seleccionados = data.get('indices', [])
        
        if len(indices_seleccionados) < 5:
            return jsonify({"error": "Selecciona al menos 5 naves industriales"}), 400
        
        inicializar_datos()
        
        # Filtrar naves seleccionadas
        df_naves_filtrado = df_naves_industriales.iloc[indices_seleccionados]
        
        # Filtrar matriz de distancias
        df_matriz_distancias = df_matriz_distancias_original.iloc[indices_seleccionados, indices_seleccionados].copy()
        df_matriz_distancias.reset_index(drop=True, inplace=True)
        df_matriz_distancias.columns = range(df_matriz_distancias.shape[1])
        
        # Realizar predicciones climáticas
        print("Realizando predicciones climáticas...")
        df_naves_filtrado = realizar_predicciones(Modelo_RandomForest, df_naves_filtrado, api_key)
        
        # Aplicar penalizaciones climáticas
        penalizaciones = {'Nublado': 1.1, 'Considerablemente nublado': 1.2, 'Despejado': 1.0,
                         'Niebla': 1.8, 'Bruma': 1.3, 'Lluvia intensa': 1.7,
                         'Tormenta electrica intensa': 1.9, 'Neblina': 1.6, 'Lluvia': 1.5, 'Truenos': 1.4}
        
        # Aplicar penalizaciones climáticas todas con valor 0 para pruebas
        penalizaciones = {clave: 1.0 for clave in penalizaciones}
        
        #Definir lambda_penalizacion
        lambda_penalizacion = 1.0
        
        for pos, fila in df_naves_filtrado.reset_index(drop=True).iterrows():
            pred = fila['Prediccion']
            if pred in penalizaciones:
                penal = penalizaciones[pred] * lambda_penalizacion
                df_matriz_distancias.iloc[pos, :] *= penal   # fila
                df_matriz_distancias.iloc[:, pos] *= penal   # columna
                df_matriz_distancias.iloc[pos, pos] = 0.0   # diagonal
        
        # Guardar matriz temporal para el algoritmo
        df_matriz_distancias.to_csv("Matriz_Distancias_Temporal.csv", header=False, index=False)
        
        # Ejecutar algoritmo de recocido simulado
        num_naves = len(df_naves_filtrado)
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        nombre_biblioteca = "recocido.dll" if os.name == 'nt' else "librecocido.so"
        ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
        
        rs = AlgoritmoRecocido(ruta_biblioteca)
        params = {'longitud_ruta': num_naves, 'num_generaciones': 800, 'tasa_enfriamiento': 0.99,
                 'temperatura_final': 0.001, 'max_neighbours': num_naves * 10, 'm': 3,
                 'nombre_archivo': "Matriz_Distancias_Temporal.csv", 'heuristica': 0}
        
        resultado = rs.ejecutar(**params)
        
        # Preparar respuesta JSON
        ruta_optimizada = []
        for idx in resultado['recorrido']:
            fila = df_naves_filtrado.iloc[idx]
            ruta_optimizada.append({
                "lat": float(fila['latitud']),
                "lng": float(fila['longitud']),
                "nombre": fila['nombre'],
                "condicion": fila['Prediccion']
            })
        
        # Limpiar archivo temporal
        if os.path.exists("Matriz_Distancias_Temporal.csv"):
            os.remove("Matriz_Distancias_Temporal.csv")
        
        return jsonify({
            "ruta": ruta_optimizada,
            "fitness": resultado['fitness'],
            "tiempo_ejecucion": resultado['tiempo_ejecucion'],
            "temperatura_inicial": resultado['temperatura_inicial'],
            "temperatura_final": resultado['temperatura_final']
        })
        
    except Exception as e:
        return jsonify({"error": f"Error al generar la ruta: {str(e)}"}), 500

def main(): # Función original para ejecución directa
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_csv_naves = os.path.join(directorio_actual, "Naves_Industriales.csv")
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

    df_matriz_distancias_original = pd.read_csv(ruta_matriz_distancias, header=None)
    
    # Filtrar la matriz para que contenga solo las filas y columnas de las naves seleccionadas
    df_matriz_distancias = df_matriz_distancias_original.iloc[indices_seleccionados, indices_seleccionados].copy()
    
    # Reiniciar los índices y nombres de columnas para que sea una matriz de 0 a N-1
    df_matriz_distancias.reset_index(drop=True, inplace=True)
    df_matriz_distancias.columns = range(df_matriz_distancias.shape[1])
    
    # Cargar modelo para predicciones climáticas
    Modelo_RandomForest = joblib.load(ruta_modelo)
    api_key = "7f25124e580c4de6a2e00312251205"
    
    inicio_tiempo = datetime.now()
    print("Realizando predicciones climáticas para las naves industriales seleccionadas...")

    df_naves_industriales_filtrado = realizar_predicciones(Modelo_RandomForest, df_naves_industriales_filtrado, api_key)

    fin_tiempo = datetime.now()
    duracion = fin_tiempo - inicio_tiempo
    print(f"Predicciones completadas en {duracion.total_seconds():.2f} segundos.")

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
    
    # Aplicar las penalizaciones
    for pos, fila in df_naves_industriales_filtrado.reset_index(drop=True).iterrows():
        pred = fila['Prediccion']
        if pred in penalizaciones:
            penal = penalizaciones[pred] * lambda_penalizacion
            # Multiplicar fila y columna usando .iloc
            df_matriz_distancias.iloc[pos, :] *= penal   # fila
            df_matriz_distancias.iloc[:, pos] *= penal   # columna
            df_matriz_distancias.iloc[pos, pos] = 0.0   # diagonal

    print("Naves Industriales disponibles para iniciar el recorrido:")
    print(df_naves_industriales_filtrado)
       
    num_naves = len(df_naves_industriales_filtrado)
    print(f"Número total de naves industriales seleccionadas: {num_naves}")
    
    # Hacer la optimizacion con el Recocido
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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Ejecutar como servidor web
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Ejecutar función main original
        main()


