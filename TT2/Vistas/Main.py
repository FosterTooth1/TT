import os
import ctypes
import joblib
import requests
import numpy as np
import pandas as pd
import mysql.connector
from functools import wraps
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import c_int, c_double, c_char_p, POINTER, Structure
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import json
import functools

PARAMETROS_PATH = os.path.join(os.path.dirname(__file__), 'parametros.json')
parametros_app = {}
df_matriz_carretera = None

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

def predecir_nave(fila, modelo, api_key, condiciones):
    lat = fila.get('latitud')
    lon = fila.get('longitud')

    if pd.isnull(lat) or pd.isnull(lon): # Validación de coordenadas
        return None
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except:
        return None
    if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
        return None
    if lat_f == 0.0 and lon_f == 0.0:
        return None

    try:
        location = f"{lat_f}, {lon_f}"
        url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
        response = requests.get(url, timeout=8)
        data = response.json()

        hora_API = data['location']['localtime']
        temperatura = data['current']['temp_c']
        puntoRocio = data['current']['dewpoint_c']
        humedad = data['current']['humidity']
        dirViento_API = data['current']['wind_dir']
        velocidadViento = data['current']['wind_kph']
    except:
        return None

    try:
        hora_num = cambiarFormatoHora(hora_API)
    except:
        return None

    dir_viento_num = cambiarFormatoViento(dirViento_API)

    columnas = ['Time', 'Temperature', 'Dew Point', 'Humidity', 'Wind', 'Wind Speed']

    fila_predict = pd.DataFrame(
        [[hora_num, temperatura, puntoRocio, humedad, dir_viento_num, velocidadViento]],
        columns=columnas
    )

    try:
        pred = modelo.predict(fila_predict)
        pred_idx = pred[0]
    except:
        return None

    if isinstance(pred_idx, (int, np.integer)) and 0 <= pred_idx < len(condiciones):
        return condiciones[pred_idx]

    return str(pred_idx)

def realizar_predicciones(modelo, df_naves_industriales_filtrado, api_key):

    condiciones = [
        'Nublado',
        'Considerablemente nublado',
        'Despejado',
        'Niebla',
        'Bruma',
        'Lluvia intensa',
        'Tormenta electrica intensa',
        'Neblina',
        'Lluvia',
        'Truenos'
    ]

    predicciones = [None] * len(df_naves_industriales_filtrado)
    with ThreadPoolExecutor(max_workers=20) as executor:

        futures = {
            executor.submit(
                predecir_nave,
                fila,
                modelo,
                api_key,
                condiciones
            ): idx
            for idx, (_, fila) in enumerate(df_naves_industriales_filtrado.iterrows())
        }

        # Recibir resultados conforme terminan
        for future in as_completed(futures):
            idx = futures[future]
            try:
                predicciones[idx] = future.result()
            except:
                predicciones[idx] = None

    # Crear DF final con predicciones
    df_resultado = df_naves_industriales_filtrado.copy()
    df_resultado['Prediccion'] = predicciones

    return df_resultado

def crear_matriz_distancias(df_naves_filtrado):
    global df_matriz_carretera
    if df_matriz_carretera is None:
        print("ADVERTENCIA: Matriz carretera no cargada, inicializando...")
        inicializar_datos()
    
    indices_originales = df_naves_filtrado.index
    
    try:
        matriz_recortada = df_matriz_carretera.iloc[indices_originales, indices_originales]
        matriz_recortada = matriz_recortada.reset_index(drop=True)
        matriz_recortada.columns = range(matriz_recortada.shape[1])
        
        return matriz_recortada
        
    except Exception as e:
        print(f"Error al crear matriz de distancias carretera: {e}")
        raise e

def inicializar_datos():
    global df_naves_industriales
    global Modelo_RandomForest
    global df_matriz_carretera
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_csv_naves = os.path.join(directorio_actual, "Naves_Industriales.csv")
    ruta_modelo = os.path.join(directorio_actual, "prediccion_clima.pkl")
    ruta_matriz = os.path.join(directorio_actual, "Matriz_Distancias_Carretera.csv")
    
    if df_naves_industriales is None:
        try:
            print("Cargando Naves Industriales...")
            df_naves_industriales = cargar_CSV(ruta_csv_naves)
            print("Cargando Modelo Random Forest...")
            Modelo_RandomForest = joblib.load(ruta_modelo)
        except Exception as e:
            print(f"Error cargando Naves o Modelo: {e}")

    if df_matriz_carretera is None:
        try:
            print(f"Intentando cargar matriz desde: {ruta_matriz}")
            df_matriz_carretera = pd.read_csv(ruta_matriz, header=None)
            print(f"Matriz de carretera cargada: {df_matriz_carretera.shape}")
        except Exception as e:
            print(f"Error cargando matriz carretera: {e}")
        
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

def calcular_distancia_real_km(recorrido_local, df_matriz_raw):
    distancia = 0.0
    try:
        for i in range(len(recorrido_local)):
            origen = recorrido_local[i]
            destino = recorrido_local[(i + 1) % len(recorrido_local)]
            distancia += df_matriz_raw.iloc[origen, destino]
    except Exception as e:
        print(f"Error calculando distancia real: {e}")
        return 0.0
    return distancia

# Esta función ejecuta la optimización y retorna la ruta procesada
def ejecutar_optimizacion(df_naves, df_matriz, params_algoritmo, indice_local_inicio, indices_globales_map):
    df_matriz.to_csv("Matriz_Distancias_Temporal.csv", header=False, index=False)
    
    num_naves = len(df_naves)
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    nombre_biblioteca = "recocido.dll" if os.name == 'nt' else "librecocido.so"
    ruta_biblioteca = os.path.join(directorio_actual, nombre_biblioteca)
    
    rs = AlgoritmoRecocido(ruta_biblioteca)
    
    resultado = rs.ejecutar(
        longitud_ruta=num_naves,
        num_generaciones=params_algoritmo['num_generaciones'],
        tasa_enfriamiento=params_algoritmo['tasa_enfriamiento'],
        temperatura_final=params_algoritmo['temperatura_final'],
        max_neighbours=num_naves * 10,
        m=params_algoritmo['m'],
        nombre_archivo="Matriz_Distancias_Temporal.csv",
        heuristica=params_algoritmo['heuristica']
    )
    
    # Rotar recorrido
    recorrido_local_ordenado = rotar_recorrido(resultado['recorrido'], indice_local_inicio)
    
    # Construir objeto ruta
    ruta_procesada = []
    indices_ordenados = []
    
    for idx_local in recorrido_local_ordenado:
        idx_global = indices_globales_map[idx_local]
        fila = df_naves.iloc[idx_local]
        
        ruta_procesada.append({
            "lat": float(fila['latitud']),
            "lng": float(fila['longitud']),
            "nombre": fila['nombre'],
            "condicion": fila.get('Prediccion', 'N/A')
        })
        indices_ordenados.append(int(idx_global))
        
    return {
        "ruta": ruta_procesada,
        "indices": indices_ordenados,
        "fitness": resultado['fitness'],
        "fitness_generaciones": resultado['fitness_generaciones'],
        "recorrido_local": recorrido_local_ordenado
    }

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

@app.route('/panel_admin')
@admin_required
def panel_admin():
    # Renderiza el template que ya existe
    return render_template('panel_admin.html', 
                           user_name=session.get('user_name', 'Admin'), 
                           params=parametros_app)

# Actualizar parámetros desde el panel de administración
@app.route('/actualizar_parametros', methods=['POST'])
@admin_required
def actualizar_parametros():
    """
    Recibe los datos del formulario del panel de admin,
    actualiza la variable global parametros_app y
    la guarda en el archivo parametros.json.
    """
    global parametros_app # Necesitamos modificar la variable global
    try:
        # 1. Actualizar los parámetros generales
        # Es crucial convertir los valores al tipo de dato correcto (int o float)
        parametros_app['lambda_penalizacion'] = float(request.form['lambda_penalizacion'])
        parametros_app['num_generaciones'] = int(request.form['num_generaciones'])
        parametros_app['tasa_enfriamiento'] = float(request.form['tasa_enfriamiento'])
        parametros_app['temperatura_final'] = float(request.form['temperatura_final'])
        parametros_app['m'] = int(request.form['m'])
        parametros_app['heuristica'] = int(request.form['heuristica'])

        # 2. Actualizar el diccionario anidado de penalizaciones
        # Iteramos sobre las claves que ya existen en el diccionario
        for key in parametros_app['penalizaciones']:
            # Recreamos el nombre del campo del formulario
            # (ej: "Considerablemente nublado" -> "penalizacion_Considerablemente_nublado")
            form_key = f"penalizacion_{key.replace(' ', '_')}"
            
            # Obtenemos el valor del formulario, lo convertimos a float y actualizamos
            if form_key in request.form:
                parametros_app['penalizaciones'][key] = float(request.form[form_key])

        # 3. Guardar los cambios en el archivo JSON
        guardar_parametros() 
        print("Parámetros actualizados y guardados por el admin.")

    except Exception as e:
        print(f"ERROR al actualizar parámetros: {e}")

    # 4. Redirigir al usuario de vuelta al panel de administración
    return redirect(url_for('panel_admin'))

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
        if not conn: return jsonify({"error": "DB Error"}), 500
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT destinos FROM Ruta WHERE id_ruta = %s AND id_usuario = %s", (ruta_id, user_id))
        ruta_db = cur.fetchone()
        cur.close()
        conn.close()
        
        if not ruta_db: return jsonify({"error": "Ruta no encontrada"}), 404
        
        indices_globales = json.loads(ruta_db['destinos'])
        if not indices_globales: return jsonify({"error": "Ruta vacía"}), 400
        
        inicializar_datos()
        df_naves_filtrado = df_naves_industriales.iloc[indices_globales]
        df_matriz_raw = crear_matriz_distancias(df_naves_filtrado) # Matriz km reales
        
        # Predicciones
        df_naves_filtrado = realizar_predicciones(Modelo_RandomForest, df_naves_filtrado, api_key)
        
        # Ruta penalizada
        df_matriz_penalizada = df_matriz_raw.copy()
        penalizaciones = parametros_app['penalizaciones']
        lambda_penal = parametros_app['lambda_penalizacion']
        
        for pos, fila in df_naves_filtrado.reset_index(drop=True).iterrows():
            pred = fila['Prediccion']
            if pred in penalizaciones:
                penal = penalizaciones[pred] * lambda_penal
                df_matriz_penalizada.iloc[pos, :] *= penal
                df_matriz_penalizada.iloc[:, pos] *= penal
                df_matriz_penalizada.iloc[pos, pos] = 0.0

        indice_local_inicio = 0
        
        # Ejecutar Algoritmo penalizado
        res_penalizada = ejecutar_optimizacion(
            df_naves_filtrado, df_matriz_penalizada, parametros_app, 
            indice_local_inicio, indices_globales
        )
        
        # Calcular Distancia Real de la ruta penalizada
        dist_real_km = calcular_distancia_real_km(res_penalizada['recorrido_local'], df_matriz_raw)

        # Ejecutar Algoritmo sin penalizaciones
        # Se ejecuta el algoritmo sobre la matriz RAW
        res_limpia = ejecutar_optimizacion(
            df_naves_filtrado, df_matriz_raw, parametros_app, 
            indice_local_inicio, indices_globales
        )

        # Limpiar archivo temporal
        if os.path.exists("Matriz_Distancias_Temporal.csv"):
            os.remove("Matriz_Distancias_Temporal.csv")

        return jsonify({
            "ruta_penalizada": res_penalizada['ruta'],
            "ruta_limpia": res_limpia['ruta'],
            "indices_penalizada": res_penalizada['indices'],
            "indices_limpia": res_limpia['indices'],
            "fitness_penalizado": res_penalizada['fitness'], 
            "distancia_real": dist_real_km,                   
            "fitness_generaciones": res_penalizada['fitness_generaciones']
        })

    except Exception as e:
        print("Error en /regenerar-ruta:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

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
        indices_globales = data.get('indices', [])
        indice_inicio_global = data.get('indice_inicio')
        
        if len(indices_globales) < 5:
            return jsonify({"error": "Selecciona al menos 5 naves"}), 400
            
        inicializar_datos()
        
        df_naves_filtrado = df_naves_industriales.iloc[indices_globales]
        try:
            indice_local_inicio = indices_globales.index(indice_inicio_global)
        except ValueError:
            return jsonify({"error": "Índice de inicio inválido"}), 400

        # Matriz base (Distancias Reales)
        df_matriz_raw = crear_matriz_distancias(df_naves_filtrado)
        
        # Predicciones
        df_naves_filtrado = realizar_predicciones(Modelo_RandomForest, df_naves_filtrado, api_key)
        
        # Optimización con penalizaciones
        df_matriz_penalizada = df_matriz_raw.copy()
        penalizaciones = parametros_app['penalizaciones']
        lambda_penal = parametros_app['lambda_penalizacion']
        
        for pos, fila in df_naves_filtrado.reset_index(drop=True).iterrows():
            pred = fila['Prediccion']
            if pred in penalizaciones:
                penal = penalizaciones[pred] * lambda_penal
                df_matriz_penalizada.iloc[pos, :] *= penal
                df_matriz_penalizada.iloc[:, pos] *= penal
                df_matriz_penalizada.iloc[pos, pos] = 0.0

        res_penalizada = ejecutar_optimizacion(
            df_naves_filtrado, df_matriz_penalizada, parametros_app, 
            indice_local_inicio, indices_globales
        )
        
        # Calcular Distancia Real de la ruta penalizada
        dist_real_km = calcular_distancia_real_km(res_penalizada['recorrido_local'], df_matriz_raw)

        # Optimización sin penalizaciones
        # Se ejecuta el algoritmo sobre la matriz RAW
        res_limpia = ejecutar_optimizacion(
            df_naves_filtrado, df_matriz_raw, parametros_app, 
            indice_local_inicio, indices_globales
        )

        if os.path.exists("Matriz_Distancias_Temporal.csv"):
            os.remove("Matriz_Distancias_Temporal.csv")

        # Se retornan ambas rutas para la comparativa
        return jsonify({
            "ruta_penalizada": res_penalizada['ruta'],
            "ruta_limpia": res_limpia['ruta'],
            "indices_penalizada": res_penalizada['indices'], # Estos son los que se guardarán
            "indices_limpia": res_limpia['indices'],
            "fitness_penalizado": res_penalizada['fitness'],
            "distancia_real": dist_real_km,
            "fitness_generaciones": res_penalizada['fitness_generaciones']
        })

    except Exception as e:
        print(f"Error generando ruta: {e}")
        return jsonify({"error": str(e)}), 500
    
###########################################################################################################################
# Eliminar Ruta de la Base de Datos
@app.route('/eliminar-ruta/<int:ruta_id>', methods=['DELETE'])
@login_required
def eliminar_ruta_endpoint(ruta_id):
    try:
        user_id = session['user_id']
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Error de conexión con la BD"}), 500
            
        cur = conn.cursor()
        
        # Ejecutar DELETE verificando que el id_ruta pertenezca al id_usuario
        cur.execute("DELETE FROM Ruta WHERE id_ruta = %s AND id_usuario = %s", (ruta_id, user_id))
        conn.commit()
        
        filas_afectadas = cur.rowcount
        cur.close()
        conn.close()
        
        if filas_afectadas > 0:
            return jsonify({"status": "ok", "message": "Ruta eliminada correctamente"}), 200
        else:
            return jsonify({"status": "error", "message": "Ruta no encontrada o no tienes permiso"}), 404
            
    except Exception as e:
        print(f"Error al eliminar ruta: {e}")
        return jsonify({"status": "error", "message": "Error interno del servidor"}), 500

###########################################################################################################################
if __name__ == "__main__":
    app.run(debug=True)
