import os
import ctypes
import joblib
import requests
import numpy as np
import pandas as pd
import mysql.connector
from functools import wraps
from datetime import datetime
from ctypes import c_int, c_double, c_char_p, POINTER, Structure
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import json
import functools

PARAMETROS_PATH = os.path.join(os.path.dirname(__file__), 'parametros.json')
parametros_app = {}

def cargar_parametros():
    """Carga los parámetros desde parametros.json a la variable global."""
    global parametros_app
    try:
        with open(PARAMETROS_PATH, 'r', encoding='utf-8') as f:
            parametros_app = json.load(f)
        print("Parámetros cargados exitosamente.")
    except Exception as e:
        print(f"ERROR AL CARGAR parámetros.json: {e}")
        # En caso de error, define valores por defecto para que la app no falle
        parametros_app = {
            "penalizaciones": {"Nublado": 1.0, "Despejado": 1.0},
            "lambda_penalizacion": 1.0, "num_generaciones": 100,
            "tasa_enfriamiento": 0.9, "temperatura_final": 0.01,
            "m": 1, "heuristica": 0
        }

def guardar_parametros():
    """Guarda la variable global de parámetros en parametros.json."""
    global parametros_app
    try:
        with open(PARAMETROS_PATH, 'w', encoding='utf-8') as f:
            json.dump(parametros_app, f, indent=2, ensure_ascii=False)
        print("Parámetros guardados exitosamente.")
    except Exception as e:
        print(f"ERROR AL GUARDAR parámetros.json: {e}")

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
        
def rotar_recorrido(recorrido_local, indice_local_inicio):
    try:
        posicion_inicio = recorrido_local.index(indice_local_inicio)
        recorrido_rotado = recorrido_local[posicion_inicio:] + recorrido_local[:posicion_inicio]
        return recorrido_rotado
    except ValueError:
        return recorrido_local
    except Exception as e:
        print(f"Error al rotar recorrido: {e}")
        return recorrido_local

# ###########################################################################################################################
# CONFIG DB - Versión solo para Cloud Run

def get_db_connection():
    """Crea una conexión a la base de datos usando el socket de Cloud SQL."""

    # Estas variables de entorno son proporcionadas por Cloud Run durante el despliegue
    db_user = os.environ.get('DB_USER')
    db_pass = os.environ.get('DB_PASS')
    db_name = os.environ.get('DB_NAME')
    db_socket_path = os.environ.get('DB_HOST') # Contiene la ruta /cloudsql/...

    # Configuración para la conexión a través del socket Unix
    conn_config = {
        "user": db_user,
        "password": db_pass,
        "database": db_name,
        "unix_socket": db_socket_path
    }

    return mysql.connector.connect(**conn_config)

###########################################################################################################################
# Configuracion de Flask
app = Flask(__name__)
app.secret_key = 'logisticlima_secret_key_2024'  # Clave secreta para sesiones, ¡¡¡¡¡ EN PRODUCCION SE DEBE DE CAMBIAR!!!!!
cargar_parametros()
df_naves_industriales = None
Modelo_RandomForest = None
api_key = os.environ.get('WEATHER_API_KEY', "7f25124e580c4de6a2e00312251205")

# Decorador para verificar autenticación
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"status": "error", "message": "Debes iniciar sesión para acceder a esta función"}), 401
        return f(*args, **kwargs)
    return decorated_function

###########################################################################################################################
# Definir las Rutas del Sistema
@app.route('/')
def index():
    user_authenticated = 'user_id' in session
    user_name = session.get('user_name', '')
    return render_template('SeleccionarNaves.html', user_authenticated=user_authenticated, user_name=user_name)

@app.route('/mejor-ruta')
def mejor_ruta():
    user_authenticated = 'user_id' in session
    user_name = session.get('user_name', '')
    return render_template('MejorRuta.html', user_authenticated=user_authenticated, user_name=user_name)

@app.route('/iniciar-sesion')
def iniciar_sesion():
    return render_template('IniciarSesion.html')

@app.route('/nueva-cuenta')
def nueva_cuenta():
    return render_template('NuevaCuenta.html')

@app.route('/rutas-recientes')
def rutas_recientes():
    user_authenticated = 'user_id' in session
    user_name = session.get('user_name', '')
    if not user_authenticated:
        return redirect(url_for('iniciar_sesion'))
    return render_template('RutasRecientes.html', user_authenticated=user_authenticated, user_name=user_name)

###########################################################################################################################
# Registro de Usuario en la Base de Datos
@app.route('/registrar_usuario', methods=['POST'])
def registrar_usuario():
    try:
        data = request.get_json(force=True)
        nombre = (data.get('nombre') or "").strip()
        correo = (data.get('email') or "").strip().lower()
        password = data.get('password') or ""

        # Realizar conexion con la base de datos
        conexion  = get_db_connection()
        if not conexion :
            return jsonify({"status": "error", "message": "Error de conexión con la base de datos."}), 500
        cur = conexion .cursor(buffered=True)

        # Verificar si el correo ya está registrado
        cur.execute("SELECT id_usuario FROM Usuario WHERE correo = %s", (correo,))
        if cur.fetchone():
            cur.close()
            conexion .close()
            return jsonify({"status": "error", "message": "El correo ya está registrado."}), 400

        # Hashear la contraseña antes de guardarla
        contraseña_hash = generate_password_hash(password)

        # Insertar el nuevo usuario
        cur.execute("INSERT INTO Usuario (nombre, correo, contraseña_hash) VALUES (%s, %s, %s)",
                    (nombre, correo, contraseña_hash))
        conexion .commit()
        nuevo_id = cur.lastrowid

        cur.close()
        conexion .close()

        # Establecer sesión automáticamente después del registro
        session['user_id'] = nuevo_id
        session['user_name'] = nombre
        session['user_email'] = correo

        return jsonify({"status": "ok", "message": f"¡Bienvenido, {nombre}!", "id": nuevo_id}), 201

    except Exception as e:
        print("Error en /registrar_usuario:", e)
        return jsonify({"status": "error", "message": "Error interno al registrar usuario."}), 500

###########################################################################################################################
# Iniciar Sesión en el Sistema
@app.route('/login_usuario', methods=['POST'])
def login_usuario():
    try:
        data = request.get_json(force=True)
        correo = (data.get('email') or "").strip().lower()
        password = data.get('password') or ""

        if not correo or not password:
            return jsonify({"status": "error", "message": "Faltan campos requeridos."}), 400

        if correo == "adminhackerpro777@gato.com" and password == "Tilin.666":
            session['user_id'] = 'admin'
            session['user_name'] = 'Administrador'
            session['user_email'] = correo
            session['is_admin'] = True
            return jsonify({
                "status": "ok", 
                "message": "Bienvenido, Admin", 
                "id": "admin", 
                "redirect_url": url_for('panel_admin')
            }), 200
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Error de conexión con la base de datos."}), 500

        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM Usuario WHERE correo = %s", (correo,))
        usuario = cur.fetchone()

        cur.close()
        conn.close()

        if not usuario:
            return jsonify({"status": "error", "message": "Correo no encontrado."}), 400

        if not check_password_hash(usuario["contraseña_hash"], password):
            return jsonify({"status": "error", "message": "Contraseña incorrecta."}), 400

        session['user_id'] = usuario["id_usuario"]
        session['user_name'] = usuario['nombre']
        session['user_email'] = usuario['correo']
        session['is_admin'] = False

        return jsonify({
            "status": "ok", 
            "message": f"Bienvenido, {usuario['nombre']}!", 
            "id": usuario["id_usuario"],
            "redirect_url": url_for('index')
        }), 200

    except Exception as e:
        print("Error en /login_usuario:", e)
        return jsonify({"status": "error", "message": "Error interno al iniciar sesión."}), 500

###########################################################################################################################
# Mostrar las Rutas del usuario en el Sistema
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin'):
            return redirect(url_for('iniciar_sesion')) # Redirige al login si no es admin
        return f(*args, **kwargs)
    return decorated_function

@app.route('/obtener_rutas', methods=['GET'])
@login_required
def obtener_rutas():
    user_id = session['user_id']
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Error de conexión con la base de datos."}), 500

        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id_ruta, destinos FROM Ruta WHERE id_usuario = %s", (user_id,))
        rutas = cur.fetchall()
        print(f"Rutas encontradas para usuario {user_id}: {len(rutas)} rutas")

        cur.close()
        conn.close()

        # Convertir los índices guardados en nombres de naves legibles
        for r in rutas:
            try:
                # Convertir string de índices a lista de enteros
                indices_string = r['destinos']
                print(f"Procesando ruta {r['id_ruta']}: indices_string='{indices_string}'")
                indices = json.loads(indices_string)
                print(f"Índices convertidos: {indices}")
                
                # Obtener nombres de las naves usando los índices
                nombres_destinos = []
                for indice in indices:
                    if 0 <= indice < len(df_naves_industriales):
                        nombre_nave = df_naves_industriales.iloc[indice]['nombre']
                        nombres_destinos.append(nombre_nave)
                
                ruta_completa_str = ""
                ruta_corta_str = ""
                
                if nombres_destinos:
                    # 1. Crear la ruta completa
                    ruta_completa_str = ' → '.join(nombres_destinos)
                    
                    # 2. Crear la ruta corta
                    if len(nombres_destinos) > 2:
                        ruta_corta_str = f"{nombres_destinos[0]} → {nombres_destinos[1]} → ... → {nombres_destinos[-1]}"
                    else:
                        # Si tiene 2 o menos naves, la corta es igual a la completa
                        ruta_corta_str = ruta_completa_str
                else:
                    ruta_completa_str = 'Ruta sin destinos'
                    ruta_corta_str = 'Ruta sin destinos'

                r['ruta_completa'] = ruta_completa_str
                r['ruta_corta'] = ruta_corta_str

                r['indices'] = indices  # Guardar también los índices para uso futuro
                print(f"Ruta corta: {r['ruta_corta']}")
            
            except Exception as e:
                print(f"Error procesando ruta {r.get('id_ruta', 'desconocida')}: {e}")
                # Manejar error en ambas variables
                r['ruta_completa'] = 'Ruta con formato inválido'
                r['ruta_corta'] = 'Ruta con formato inválido'
                r['indices'] = []

        return jsonify({"status": "ok", "rutas": rutas}), 200

    except Exception as e:
        print("Error en /obtener_rutas:", e)
        return jsonify({"status": "error", "message": "Error al obtener rutas."}), 500

###########################################################################################################################
# Regenerar ruta desde índices guardados
@app.route('/regenerar-ruta/<int:ruta_id>', methods=['GET'])
@login_required
def regenerar_ruta(ruta_id):
    try:
        user_id = session['user_id']
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Error de conexión con la base de datos."}), 500
        
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT destinos FROM Ruta WHERE id_ruta = %s AND id_usuario = %s", (ruta_id, user_id))
        ruta = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if not ruta:
            return jsonify({"status": "error", "message": "Ruta no encontrada"}), 404
        
        # Obtener los índices guardados (ya están en el orden deseado)
        indices_string = ruta['destinos']
        indices_globales_guardados = json.loads(indices_string)

        if not indices_globales_guardados:
            return jsonify({"status": "error", "message": "Ruta guardada está vacía"}), 400

        # Obtener el índice global de la nave de inicio
        indice_local_inicio = 0
        
        # Filtrar las naves usando los índices
        df_naves_filtrado = df_naves_industriales.iloc[indices_globales_guardados]
        
        # Crear matriz de distancias
        df_matriz_distancias = crear_matriz_distancias(df_naves_filtrado)
        
        # Realizar predicciones climáticas
        df_naves_filtrado = realizar_predicciones(Modelo_RandomForest, df_naves_filtrado, api_key)
        
        # Aplicar penalizaciones climáticas
        global parametros_app # Accede a los parámetros cargados
        penalizaciones = parametros_app['penalizaciones']
        lambda_penalizacion = parametros_app['lambda_penalizacion']
        for pos, fila in df_naves_filtrado.reset_index(drop=True).iterrows():
            pred = fila['Prediccion']
            if pred in penalizaciones:
                penal = penalizaciones[pred] * lambda_penalizacion
                df_matriz_distancias.iloc[pos, :] *= penal
                df_matriz_distancias.iloc[:, pos] *= penal
                df_matriz_distancias.iloc[pos, pos] = 0.0

        df_matriz_distancias.to_csv("Matriz_Distancias_Temporal.csv", header=False, index=False)

        # Ejecutar optimización
        num_naves = len(df_naves_filtrado)
        directorio_actual = os.path.dirname(os.path.abspath(__file__))
        nombre_biblioteca = "recocido.dll" if os.name == 'nt' else "librecocido.so"
        ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
        rs = AlgoritmoRecocido(ruta_biblioteca)
        params = {
            'longitud_ruta': num_naves, 
            'num_generaciones': parametros_app['num_generaciones'], 
            'tasa_enfriamiento': parametros_app['tasa_enfriamiento'],
            'temperatura_final': parametros_app['temperatura_final'], 
            'max_neighbours': num_naves * 10,
            'm': parametros_app['m'],
            'nombre_archivo': "Matriz_Distancias_Temporal.csv", 
            'heuristica': parametros_app['heuristica']
        }
        resultado = rs.ejecutar(**params)
        
        # Rotar recorrido para iniciar desde la nave seleccionada
        recorrido_local_ordenado = rotar_recorrido(resultado['recorrido'], indice_local_inicio)

        # Construir ruta optimizada
        ruta_optimizada = []
        indices_globales_ordenados = []
        for idx_local in recorrido_local_ordenado:
            # Mapear el índice local (del df_filtrado) al índice global original
            idx_global = indices_globales_guardados[idx_local] 
            
            # Obtener la fila del dataframe filtrado
            fila = df_naves_filtrado.iloc[idx_local]
            
            ruta_optimizada.append({"lat": float(fila['latitud']),
                                    "lng": float(fila['longitud']),
                                    "nombre": fila['nombre'],
                                    "condicion": fila['Prediccion']})
            
            # Guardar el índice global correspondiente
            indices_globales_ordenados.append(int(idx_global)) 

        # Limpiar archivo temporal
        if os.path.exists("Matriz_Distancias_Temporal.csv"):
            os.remove("Matriz_Distancias_Temporal.csv")

        return jsonify({"ruta": ruta_optimizada,
            "indices": indices_globales_ordenados,
            "fitness": resultado['fitness'],
            "tiempo_ejecucion": resultado['tiempo_ejecucion'],
            "temperatura_inicial": resultado['temperatura_inicial'],
            "temperatura_final": resultado['temperatura_final']})

    except Exception as e:
        print("Error en /regenerar-ruta:", e)
        return jsonify({"status": "error", "message": "Error al regenerar la ruta."}), 500

###########################################################################################################################
# Cerrar Sesión
@app.route('/cerrar-sesion', methods=['POST'])
def cerrar_sesion():
    session.clear()
    return jsonify({"status": "ok", "message": "Sesión cerrada correctamente"}), 200

###########################################################################################################################
# Guardar Ruta 
@app.route('/guardar-ruta', methods=['POST'])
@login_required
def guardar_ruta():
    try:
        data = request.get_json()
        destinos = data.get('destinos', [])
        indices = data.get('indices', []) 
        user_id = session['user_id']

        print(f"Guardando ruta para usuario {user_id}")
        print(f"Destinos recibidos: {len(destinos)} | Índices recibidos: {indices}")

        # Validaciones básicas
        if not destinos:
            return jsonify({"status": "error", "message": "No hay destinos para guardar"}), 400
        if not indices or len(indices) != len(destinos):
            return jsonify({"status": "error", "message": "Los índices no coinciden con los destinos"}), 400

        # Asegurarse de que user_id sea entero
        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({"status": "error", "message": "ID de usuario inválido"}), 400

        # Crear string de índices
        indices_string = json.dumps(indices)
        print(f"Índices finales para insertar: {indices_string}")

        # Verificar longitud antes de insertar
        if len(indices_string) > 1000:
            return jsonify({"status": "error", "message": "Ruta demasiado larga para almacenarse"}), 400

        # Conexión con la base de datos
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Error de conexión con la base de datos."}), 500
        
        cur = conn.cursor()

        # Insertar ruta en la base de datos
        print(f"Insertando: user_id={user_id}, destinos='{indices_string}'")
        cur.execute("INSERT INTO Ruta (id_usuario, destinos) VALUES (%s, %s)", 
                    (user_id, indices_string))
        conn.commit()

        ruta_id = cur.lastrowid
        print(f"Ruta insertada correctamente con ID {ruta_id}")

        cur.close()
        conn.close()

        return jsonify({"status": "ok", "message": "Ruta guardada correctamente", "ruta_id": ruta_id}), 200

    except Exception as e:
        print(f"Error al guardar la ruta: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error interno al guardar la ruta",
            "exception": str(e)
        }), 500

###########################################################################################################################
# Obtener naves para la Optimizacion
@app.route('/api/naves', methods=['GET'])
def obtener_naves():
    inicializar_datos()
    if df_naves_industriales is None:
        return jsonify({"error": "No se pudieron cargar las naves industriales"}), 500
    naves = df_naves_industriales[['nombre', 'latitud', 'longitud']].to_dict('records')
    return jsonify(naves)

###########################################################################################################################
# Realizar la Optimizacion
@app.route('/api/generar-ruta', methods=['POST'])
def generar_ruta():
    try:
        data = request.get_json()
        indices_seleccionados = data.get('indices', [])
        indice_inicio_global = data.get('indice_inicio')
        if len(indices_seleccionados) < 5:
            return jsonify({"error": "Selecciona al menos 5 naves industriales"}), 400
        inicializar_datos()
        if indice_inicio_global is None or indice_inicio_global not in indices_seleccionados:
            return jsonify({"error": "El índice de inicio no es válido o no está en la lista de seleccionados"}), 400
        df_naves_filtrado = df_naves_industriales.iloc[indices_seleccionados]
        try:
            indice_local_inicio = indices_seleccionados.index(indice_inicio_global)
        except ValueError:
            return jsonify({"error": "Error interno al mapear el índice de inicio"}), 500
        df_matriz_distancias = crear_matriz_distancias(df_naves_filtrado)
        
        df_naves_filtrado = realizar_predicciones(Modelo_RandomForest, df_naves_filtrado, api_key)
        
        global parametros_app # Accede a los parámetros cargados
        penalizaciones = parametros_app['penalizaciones']
        lambda_penalizacion = parametros_app['lambda_penalizacion']
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
        params = {
            'longitud_ruta': num_naves, 
            'num_generaciones': parametros_app['num_generaciones'], 
            'tasa_enfriamiento': parametros_app['tasa_enfriamiento'],
            'temperatura_final': parametros_app['temperatura_final'], 
            'max_neighbours': num_naves * 10,
            'm': parametros_app['m'],
            'nombre_archivo': "Matriz_Distancias_Temporal.csv", 
            'heuristica': parametros_app['heuristica']
        }
        resultado = rs.ejecutar(**params)
        
        recorrido_local_ordenado = rotar_recorrido(resultado['recorrido'], indice_local_inicio)

        ruta_optimizada = []
        
        indices_globales_ordenados = []
        
        for idx_local in recorrido_local_ordenado:
            # Mapear el índice local (del df_filtrado) al índice global original
            idx_global = indices_seleccionados[idx_local]
            
            # Obtener la fila del dataframe filtrado
            fila = df_naves_filtrado.iloc[idx_local]
            
            ruta_optimizada.append({"lat": float(fila['latitud']),
                                    "lng": float(fila['longitud']),
                                    "nombre": fila['nombre'],
                                    "condicion": fila['Prediccion']})
            
            # Guardar el índice global correspondiente
            indices_globales_ordenados.append(int(idx_global)) # Asegurar que sea int

        if os.path.exists("Matriz_Distancias_Temporal.csv"):
            os.remove("Matriz_Distancias_Temporal.csv")

        return jsonify({"ruta": ruta_optimizada,
                        "indices": indices_globales_ordenados,
                        "fitness": resultado['fitness'],
                        "tiempo_ejecucion": resultado['tiempo_ejecucion'],
                        "temperatura_inicial": resultado['temperatura_inicial'],
                        "temperatura_final": resultado['temperatura_final']})

    except Exception as e:
        return jsonify({"error": f"Error al generar la ruta: {str(e)}"}), 500

###########################################################################################################################
if __name__ == "__main__":
    app.run(debug=True)
