import os
import pickle
import joblib
import re
import string
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from langchain_community.retrievers import TFIDFRetriever

from langchain_core.documents import Document 

print("Iniciando Script de Creación de TF-IDF")

try:
    print("Descargando recursos de NLTK (stopwords, punkt)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    # Configurar para español
    stop_words_es = set(stopwords.words('spanish'))
    stemmer_es = SnowballStemmer('spanish')
    
    # Añadir puntuación española que string.punctuation puede omitir
    custom_punctuation = string.punctuation + '¿¡'
    # Crear el 'translator' para eliminar puntuación eficientemente
    translator = str.maketrans('', '', custom_punctuation)

    print("Recursos de NLTK listos.")
except Exception as e:
    print(f"Error al descargar/configurar NLTK. Verifica tu conexión a internet. Error: {e}")
    exit()

# Rutas de archivos
CHILD_DOCS_CACHE_PATH = 'child_docs_cache.pkl' 
TFIDF_PATH = 'tfidf_retriever.joblib'


# Función de Limpieza y Preprocesamiento para Español
def preprocess_spanish_text(text: str) -> str:
    # 1. Minúsculas y Regex
    text = text.lower()
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    text = re.sub(r'<.*?>', '', text)  # HTML
    text = re.sub(r'\S+@\S+', '', text)  # Emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # URLs
    
    # 2. Eliminar números
    text = re.sub(r'\d+', '', text)
    
    # 3. Eliminar puntuación (usando translator)
    text = text.translate(translator)
    
    # 4. Tokenizar
    word_tokens = word_tokenize(text, language='spanish')
    
    # 5. Eliminar stopwords y aplicar stemming
    # Eliminar palabras de 2 carácteres también
    processed_words = [
        stemmer_es.stem(word) 
        for word in word_tokens 
        if word not in stop_words_es and len(word) > 1
    ]
    
    # Unir de nuevo a un string
    return ' '.join(processed_words)


# 1. Eliminar el índice TF-IDF antiguo si existe
if os.path.exists(TFIDF_PATH):
    print(f"Eliminando índice TF-IDF antiguo en {TFIDF_PATH}...")
    os.remove(TFIDF_PATH)

# 2. Verificar que el archivo cache de los chunks hijos exista
if not os.path.exists(CHILD_DOCS_CACHE_PATH):
    print(f"ERROR: No se encontró el archivo '{CHILD_DOCS_CACHE_PATH}'.")
    print("Asegúrate de ejecutar primero el script de ingestión principal.")
    exit()

# 3. Cargar los chunks hijos desde el archivo cache
try:
    print(f"Cargando chunks HIJO desde {CHILD_DOCS_CACHE_PATH}...")
    with open(CHILD_DOCS_CACHE_PATH, "rb") as f:
        all_child_docs = pickle.load(f)
    
    if not all_child_docs:
        print("ERROR: El archivo de cache está vacío. No se pueden crear los índices.")
        exit()
        
    print(f"Se cargaron {len(all_child_docs)} chunks HIJO.")

except Exception as e:
    print(f"Error al cargar el archivo '{CHILD_DOCS_CACHE_PATH}': {e}")
    exit()

# 4. Aplicar el preprocesamiento a los documentos cargados
print("Iniciando preprocesamiento (limpieza, stemming) del texto...")

try:
    for i, doc in enumerate(all_child_docs):
        # 1. Guardar el texto original
        original_text = doc.page_content
        
        # 2. Procesar el texto
        processed_text = preprocess_spanish_text(original_text)
        
        # 3. Reemplazar el contenido del documento
        doc.page_content = processed_text
        
        if (i + 1) % 100 == 0 or (i + 1) == len(all_child_docs):
             print(f"Preprocesados {i+1}/{len(all_child_docs)} documentos")
             
    print("Preprocesamiento completado.")
except Exception as e:
    print(f"Error durante el preprocesamiento del texto: {e}")
    exit()


# 4. Crear y guardar el TF-IDF Retriever
print(f"\nCreando el Índice TF-IDF con el texto preprocesado...")

tfidf_retriever = TFIDFRetriever.from_documents(
    documents=all_child_docs,
)

print(f"Guardando el retriever TF-IDF en: {TFIDF_PATH}...")
joblib.dump(tfidf_retriever, TFIDF_PATH)

print(f"Proceso completado. Índice TF-IDF guardado en '{TFIDF_PATH}'")