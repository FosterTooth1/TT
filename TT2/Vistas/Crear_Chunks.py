import os
import shutil
import hashlib
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.stores import InMemoryStore
import random
import pickle
import uuid


import tiktoken
print(tiktoken.__version__)

# Cargar variables de entorno
print("Cargando variables de entorno...")
load_dotenv()

OpenAI_api_key = os.getenv("OPENAI_API_KEY")
if not OpenAI_api_key:
    print("ERROR: La API Key de OpenAI no se encontró. Asegúrate de tener un archivo .env con OPENAI_API_KEY='sk-...'")
    exit()

# Rutas de los documentos, bases de datos y el tfidf
CHROMA_PATH = 'chroma_child_db' 
DOCSTORE_PATH = 'parent_docstore.pkl'
CHILD_DOCS_CACHE_PATH = 'child_docs_cache.pkl'

# Inicializar el modelo de OpenAI para chat y embeddings
print("Inicializando modelos de OpenAI...")

# Modelo para embeddings (chunking y almacenamiento)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    openai_api_key=OpenAI_api_key
)

# Modelo para contextualización
chat_model = ChatOpenAI(
    model="gpt-5-nano",
    openai_api_key=OpenAI_api_key,
    verbosity="low"
)

# Inicializar parent y child splitters
print("Inicializando Text Splitters...")

# El Parent Splitter divide el documento en secciones más grandes (contexto)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=200,
    separators="\n\n, \n"
)

# El Child Splitter divide los chunks padre en piezas más pequeñas (para embedding)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators="\n"
)

# Eliminar la base de datos Chroma antigua si existe
if os.path.exists(CHROMA_PATH):
    print(f"Eliminando DB antigua en {CHROMA_PATH}...")
    shutil.rmtree(CHROMA_PATH)
    
# Eliminar el cache de chunks hijos antiguo si existe
if os.path.exists(CHILD_DOCS_CACHE_PATH):
    print(f"Eliminando cache de chunks hijos antiguo en {CHILD_DOCS_CACHE_PATH}...")
    os.remove(CHILD_DOCS_CACHE_PATH)

# El docstore guardará los PADRES (Contextualizados)
docstore = InMemoryStore() 

# El vectorstore guardará los HIJOS (Originales, para búsqueda)
vectorstore = Chroma(
    collection_name="child_docs_v1",
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH
)

# Lista para guardar los chunks hijos antes de añadirlos en lote
all_child_docs_to_add = []

# Obtenemos el archivo manual .txt
text_file = 'Manual.txt'
if not text_file:
    print(f"ERROR: No se encontró el archivo 'Manual.txt' en el directorio actual.")
    exit()
print(f"Encontrados {len(text_file)} documentos Markdown para procesar...")

# Se procesa cada archivo markdown individualmente
print(f"\nProcesando: {text_file}")
file_path = os.path.join(os.getcwd(), text_file)

try:
    # Cargar el contenido completo del documento
    with open(file_path, 'r', encoding='utf-8') as f:
        whole_document_text = f.read()
        
        # Crear un objeto 'Document' de LangChain para el splitter
        # Se pasa el nombre del archivo en los metadatos para rastreo
        source_doc = Document(
            page_content=whole_document_text,
            metadata={"source": text_file}
        )

        # Se divide el documento completo en chunks PADRE (secciones)
        print("Dividiendo en chunks PADRE (secciones)...")
        parent_chunks = parent_splitter.split_documents([source_doc])
        print(f"Documento dividido en {len(parent_chunks)} chunks PADRE.")

        # Preparar el Prompt Caching (usando el DOCUMENTO ENTERO como contexto)
        static_prefix = f"""
<document>
{whole_document_text}
</document>
Basado en el documento completo de arriba, tu tarea es proporcionar contexto para el fragmento (chunk) que te daré a continuación.
El contexto debe ser breve y conciso, diseñado para ubicar el fragmento dentro del documento general.
Responde únicamente con el contexto conciso.
Identifica dentro del contexto el Capítulo al que pertenece.

Formato de salida (exacto):
    [Contexto conciso]
    (Capítulo)
    
Ejemplo:
    Dentro de la pestaña "Guardar Rutas", el usuario puede ver todas las rutas creadas previamente.
    (Guardar Rutas)
    
Aquí está el fragmento que queremos ubicar:
<chunk>
"""
        doc_cache_key = hashlib.md5(static_prefix.encode()).hexdigest()

        # Iterar sobre cada chunk PADRE para procesarlo
        print(f"Contextualizando {len(parent_chunks)} chunks PADRE y dividiendo en HIJOS...")
        for i, parent_chunk in enumerate(parent_chunks):
            original_parent_content = parent_chunk.page_content
            
            # Contextualizar el chunk PADRE usando el modelo de chat
            dynamic_suffix = f"\n{original_parent_content}\n</chunk>"
            full_prompt = static_prefix + dynamic_suffix
            
            response = chat_model.invoke(
                [HumanMessage(content=full_prompt)],
                prompt_cache_key=doc_cache_key
            )
            generated_context = response.content.strip()
            
            # Este es el contenido final que se guardará en el docstore
            contextualized_parent_text = f"{generated_context}\n{original_parent_content}"

            # Guardar el chunk PADRE contextualizado
            parent_id = str(uuid.uuid4())
            new_parent_doc = Document(
                page_content=contextualized_parent_text,
                metadata=parent_chunk.metadata # Hereda {"source": "filename.txt"}
            )
            
            # Guardar el padre contextualizado en el docstore
            docstore.mset([(parent_id, new_parent_doc)])

            # Se divide el chunk PADRE (el original) en chunks HIJO
            # Se usa .create_documents para no perder la metadata de la fuente
            child_chunks = child_splitter.create_documents(
                [original_parent_content], 
                metadatas=[parent_chunk.metadata]
            )
            
            for child_chunk in child_chunks:
                # Se añade la referencia al ID del padre en la metadata del hijo
                child_chunk.metadata["parent_id"] = parent_id
                all_child_docs_to_add.append(child_chunk)
                
            if (i + 1) % 5 == 0 or (i + 1) == len(parent_chunks):
                print(f"    ... Padre {i+1}/{len(parent_chunks)} procesado (generó {len(child_chunks)} hijos).")
except Exception as e:
    print(f"ERROR al procesar el archivo {text_file}: {e}")
    # Si ocurre un error, puedes manejarlo aquí o simplemente pasar
    pass

if not all_child_docs_to_add:
    print("\nNo se generaron chunks. Terminando.")
    exit()

print(f"\nGuardando en Base de Datos")

# Guardar los CHUNKS HIJO en Chroma
print(f"Añadiendo {len(all_child_docs_to_add)} chunks HIJO a Chroma en {CHROMA_PATH}...")
vectorstore.add_documents(all_child_docs_to_add)
print("Base de datos Chroma (hijos) creada y persistida.")

# Guardar el DOCSTORE (padres) en un archivo
num_parent_docs = len(list(docstore.yield_keys()))
print(f"Guardando {num_parent_docs} chunks PADRE (contextualizados) en {DOCSTORE_PATH}...")
with open(DOCSTORE_PATH, "wb") as f:
    pickle.dump(docstore, f)
print("Docstore (padres) guardado.")

# Guardar la lista de chunks HIJO para el script TF-IDF
print(f"Guardando {len(all_child_docs_to_add)} chunks HIJO (para TF-IDF) en {CHILD_DOCS_CACHE_PATH}...")
with open(CHILD_DOCS_CACHE_PATH, "wb") as f:
    pickle.dump(all_child_docs_to_add, f)
print("Cache de chunks hijos guardado.")

# Imprimir 10 chunks aleatorios para ver cómo quedaron (se muestran los hijos)
print("\nMostrando 10 ejemplos aleatorios de chunks HIJO (lo que se guardó en Chroma):")
total_chunks_hijo = len(all_child_docs_to_add)
num_samples = min(10, total_chunks_hijo)

if num_samples > 0:
    random_samples = random.sample(all_child_docs_to_add, num_samples)
    for i, example_doc in enumerate(random_samples):
        print(f"Muestra Aleatoria {i+1}/{num_samples}")
        print(f"(Fuente: {example_doc.metadata.get('source', 'N/A')})")
        print(f"(ID Padre: {example_doc.metadata.get('parent_id', 'N/A')})")
        print(example_doc.page_content)
        print("-----\n")
else:
    print("No hay chunks hijos disponibles para mostrar.")

