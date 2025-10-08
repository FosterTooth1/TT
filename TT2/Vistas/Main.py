import os
import ctypes
import joblib
import requests
import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from ctypes import c_int, c_double, c_char_p, POINTER, Structure
from werkzeug.security import generate_password_hash, check_password_hash

###########################################################################################################################
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

###########################################################################################################################
def cargar_CSV(nombre_archivo):
    df = pd.read_csv(nombre_archivo)
    return df
    
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
    for idx, fila in df_naves_industriales_filtrado.iterrows():
        lat = fila.get('latitud')
        lon = fila.get('longitud')
        if pd.isnull(lat) or pd.isnull(lon):
            predicciones.append(None)
            continue
        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except:
            predicciones.append(None)
            continue
        if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
            predicciones.append(None)
            continue
        if lat_f == 0.0 and lon_f == 0.0:
            predicciones.append(None)
            continue

        location = f"{lat_f}, {lon_f}"
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            hora_API = data['location']['localtime']
            temperatura = data['current']['temp_c']
            puntoRocio = data['current']['dewpoint_c']
            humedad = data['current']['humidity']
            dirViento_API = data['current']['wind_dir']
            velocidadViento = data['current']['wind_kph']
        except:
            predicciones.append(None)
            continue

        try:
            hora_num = cambiarFormatoHora(hora_API)
        except:
            predicciones.append(None)
            continue

        dir_viento_num = cambiarFormatoViento(dirViento_API)

        columnas = ['Time', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed']
        fila_predict = pd.DataFrame([[hora_num, temperatura, puntoRocio, humedad, dir_viento_num, velocidadViento]], columns=columnas)

        try:
            pred_array = modelo.predict(fila_predict)
            pred_idx = pred_array[0] if len(pred_array) > 0 else None
        except:
            pred_idx = None

        etiqueta = condiciones[pred_idx] if isinstance(pred_idx, (int, np.integer)) and 0 <= pred_idx < len(condiciones) else str(pred_idx)
        predicciones.append(etiqueta)

    df = df_naves_industriales_filtrado.copy()
    df.loc[:, 'Prediccion'] = predicciones
    return df

def crear_matriz_distancias(df_naves_industriales):
    lats = df_naves_industriales['latitud'].astype(float).values
    lons = df_naves_industriales['longitud'].astype(float).values

    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    lat1 = lat_rad[:, np.newaxis]
    lat2 = lat_rad[np.newaxis, :]
    lon1 = lon_rad[:, np.newaxis]
    lon2 = lon_rad[np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))

    R = 6371.0
    dist_matrix = R * c

    df_dist = pd.DataFrame(dist_matrix)
    return df_dist

def inicializar_datos():
    global df_naves_industriales, df_matriz_distancias_original, Modelo_RandomForest
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_csv_naves = os.path.join(directorio_actual, "Naves_Industriales.csv")
    ruta_modelo = os.path.join(directorio_actual, "prediccion_clima.pkl")
    
    if df_naves_industriales is None:
        df_naves_industriales = cargar_CSV(ruta_csv_naves)
        Modelo_RandomForest = joblib.load(ruta_modelo)

###########################################################################################################################
# --- CONFIG DB (ajusta estos valores) ---
DB_CONFIG = {"host": "localhost",
             "user": "root",
             "password": "Sergio2",
             "database": "BDD_LogistiClima"}

def get_db_connection():
    """Devuelve una nueva conexión a la BD."""
    return mysql.connector.connect(**DB_CONFIG)

###########################################################################################################################
# Configuracion de Flask
app = Flask(__name__)

df_naves_industriales = None
Modelo_RandomForest = None
api_key = "7f25124e580c4de6a2e00312251205"

@app.route('/')
def index():
    return render_template('SeleccionarNaves.html')

@app.route('/mejor-ruta')
def mejor_ruta():
    return render_template('MejorRuta.html')

@app.route('/iniciar-sesion')
def iniciar_sesion():
    return render_template('IniciarSesion.html')

@app.route('/nueva-cuenta')
def nueva_cuenta():
    return render_template('NuevaCuenta.html')

@app.route('/rutas-recientes')
def rutas_recientes():
    return render_template('RutasRecientes.html')

@app.route('/api/naves', methods=['GET'])
def obtener_naves():
    inicializar_datos()
    if df_naves_industriales is None:
        return jsonify({"error": "No se pudieron cargar las naves industriales"}), 500
    naves = df_naves_industriales[['nombre', 'latitud', 'longitud']].to_dict('records')
    return jsonify(naves)

###########################################################################################################################
@app.route('/api/generar-ruta', methods=['POST'])
def generar_ruta():
    try:
        data = request.get_json()
        indices_seleccionados = data.get('indices', [])
        if len(indices_seleccionados) < 5:
            return jsonify({"error": "Selecciona al menos 5 naves industriales"}), 400
        inicializar_datos()
        df_naves_filtrado = df_naves_industriales.iloc[indices_seleccionados]
        df_matriz_distancias = crear_matriz_distancias(df_naves_filtrado)
        
        df_naves_filtrado = realizar_predicciones(Modelo_RandomForest, df_naves_filtrado, api_key)
        
        penalizaciones = {clave: 1.0 for clave in ['Nublado','Considerablemente nublado','Despejado','Niebla','Bruma',
                                                  'Lluvia intensa','Tormenta electrica intensa','Neblina','Lluvia','Truenos']}
        lambda_penalizacion = 1.0
        for pos, fila in df_naves_filtrado.reset_index(drop=True).iterrows():
            pred = fila['Prediccion']
            if pred in penalizaciones:
                penal = penalizaciones[pred] * lambda_penalizacion
                df_matriz_distancias.iloc[pos, :] *= penal
                df_matriz_distancias.iloc[:, pos] *= penal
                df_matriz_distancias.iloc[pos, pos] = 0.0

        df_matriz_distancias.to_csv("Matriz_Distancias_Temporal.csv", header=False, index=False)

        num_naves = len(df_naves_filtrado)
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        nombre_biblioteca = "recocido.dll" if os.name == 'nt' else "librecocido.so"
        ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
        rs = AlgoritmoRecocido(ruta_biblioteca)
        params = {'longitud_ruta': num_naves, 'num_generaciones': 800, 'tasa_enfriamiento': 0.99,
                  'temperatura_final': 0.001, 'max_neighbours': num_naves * 10, 'm': 3,
                  'nombre_archivo': "Matriz_Distancias_Temporal.csv", 'heuristica': 0}
        resultado = rs.ejecutar(**params)

        ruta_optimizada = []
        for idx in resultado['recorrido']:
            fila = df_naves_filtrado.iloc[idx]
            ruta_optimizada.append({"lat": float(fila['latitud']),
                                    "lng": float(fila['longitud']),
                                    "nombre": fila['nombre'],
                                    "condicion": fila['Prediccion']})

        if os.path.exists("Matriz_Distancias_Temporal.csv"):
            os.remove("Matriz_Distancias_Temporal.csv")

        return jsonify({"ruta": ruta_optimizada,
            "fitness": resultado['fitness'],
            "tiempo_ejecucion": resultado['tiempo_ejecucion'],
            "temperatura_inicial": resultado['temperatura_inicial'],
            "temperatura_final": resultado['temperatura_final']})

    except Exception as e:
        return jsonify({"error": f"Error al generar la ruta: {str(e)}"}), 500

###########################################################################################################################
if __name__ == "__main__":
    app.run(debug=True)
