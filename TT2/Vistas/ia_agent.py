import os
import json
import pickle
import joblib
import re
import string
import unicodedata
import nltk
from typing import TypedDict, Annotated, List

# --- Imports de LangChain y AI ---
import mysql.connector
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Imports específicos para tu RAG Híbrido
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer


# Configuración de Rutas
CHROMA_PATH = 'chroma_child_db'
DOCSTORE_PATH = 'parent_docstore.pkl'
TFIDF_PATH = 'tfidf_retriever.joblib'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configuración de NLTK
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    stop_words_es = set(stopwords.words('spanish'))
    stemmer_es = SnowballStemmer('spanish')
    custom_punctuation = string.punctuation + '¿¡'
    translator = str.maketrans('', '', custom_punctuation)
except Exception as e:
    print(f"Advertencia NLTK: {e}")

# Función de Preprocesamiento
def preprocess_spanish_text(text: str) -> str:
    text = text.lower()
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = text.translate(translator)
    word_tokens = word_tokenize(text, language='spanish')
    processed_words = [
        stemmer_es.stem(word) 
        for word in word_tokens 
        if word not in stop_words_es and len(word) > 1
    ]
    return ' '.join(processed_words)

# Funciones Auxiliares de Documentos

# Esta función aplana listas de listas de Documentos y elimina duplicados
def flatten_and_deduplicate_docs(list_of_lists: list[list[Document]]) -> list[Document]:
    unique_docs = {}
    for doc_list in list_of_lists:
        for doc in doc_list:
            identifier = (doc.page_content, doc.metadata.get('source'))
            if identifier not in unique_docs:
                unique_docs[identifier] = doc
    return list(unique_docs.values())

# Esta función recupera los documentos PADRE desde el docstore dado una lista de documentos HIJO
def fetch_parent_documents(child_docs: list[Document]) -> list[Document]:
    unique_parent_ids = set()
    for doc in child_docs:
        if "parent_id" in doc.metadata:
            unique_parent_ids.add(doc.metadata["parent_id"])
    parent_docs = loaded_docstore.mget(list(unique_parent_ids))
    return [doc for doc in parent_docs if doc is not None]

print("Cargando cerebro del RAG Híbrido...")
try:
    # Modelo de Embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
    
    # Parent Docstore
    with open(DOCSTORE_PATH, "rb") as f:
        loaded_docstore = pickle.load(f)

    # Retrievers (Chroma + TF-IDF)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model, collection_name="child_docs_v1")
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    tfidf_retriever = joblib.load(TFIDF_PATH)
    tfidf_retriever.k = 10

    # Reranker (CrossEncoder)
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base", model_kwargs={'device': 'cpu'}) 
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3) # Top 3 documentos más relevantes

    # Cadenas (LCEL)
    
    # Query Expansion (Reescritura)
    rewrite_llm = ChatOpenAI(model="gpt-5-nano", reasoning={ "effort": "low" }, 
                       text={ "verbosity": "low" }, openai_api_key=OPENAI_API_KEY)
    REWRITE_PROMPT = """Actúas como una herramienta de expansión de consultas. 
    Tu objetivo es generar 3 versiones diferentes de la pregunta del usuario para mejorar la búsqueda en la base de conocimientos.
    Pregunta original: {question}
    Salida (separada por saltos de línea):"""
    rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
    
    expansion_chain = (
        rewrite_prompt | rewrite_llm | StrOutputParser() | 
        RunnableLambda(lambda s: [q.strip() for q in s.strip().split('\n') if q.strip()])
    )

    # Recuperación Híbrida
    def retrieve_hybrid(query: str) -> list[Document]:
        semantic_docs = chroma_retriever.invoke(query)
        processed_query = preprocess_spanish_text(query)
        lexical_docs = tfidf_retriever.invoke(processed_query)
        return semantic_docs + lexical_docs

    hybrid_retrieval_chain = RunnableLambda(
        lambda queries: flatten_and_deduplicate_docs([retrieve_hybrid(q) for q in queries])
    )

    # Cadena Final de Recuperación
    rag_chain = (
        {
            "docs_to_rerank": expansion_chain | hybrid_retrieval_chain,
            "original_query": RunnablePassthrough()
        }
        | RunnableLambda(lambda x: compressor.compress_documents(x["docs_to_rerank"], x["original_query"]))
        | RunnableLambda(fetch_parent_documents)
    )
    
    print("RAG Híbrido cargado exitosamente.")

except Exception as e:
    print(f"CRITICAL ERROR cargando RAG: {e}")
    rag_chain = None # Fallback si falla la carga

# HERRAMIENTAS DEL AGENTE

def get_db_connection_agent():
    return mysql.connector.connect(
        user=os.environ.get('DB_USER'),
        password=os.environ.get('DB_PASS'),
        database=os.environ.get('DB_NAME'),
        unix_socket=os.environ.get('DB_HOST')
    )

@tool
def eliminar_ruta_por_posicion(posicion_visual: int, ids_en_pantalla: List[int], user_id: int):
    """
    Elimina una ruta basada en su posición visual en la tabla que ve el usuario (1, 2, 3...).
    Args:
        posicion_visual: El número de fila que ve el usuario (Por ejemplo: "Elimina la segunda ruta" -> 2).
        ids_en_pantalla: La lista de IDs reales de ruta que están actualmente renderizados en el frontend.
        user_id: El ID del usuario actual.
    """
    try:
        if posicion_visual < 1 or posicion_visual > len(ids_en_pantalla):
            return "Error: La posición indicada no existe en la pantalla actual."
        
        id_ruta_real = ids_en_pantalla[posicion_visual - 1]
        
        conn = get_db_connection_agent()
        cur = conn.cursor()
        cur.execute("DELETE FROM Ruta WHERE id_ruta = %s AND id_usuario = %s", (id_ruta_real, user_id))
        conn.commit()
        rows = cur.rowcount
        cur.close()
        conn.close()
        
        if rows > 0:
            return f"Ruta en posición {posicion_visual} (ID {id_ruta_real}) eliminada correctamente."
        else:
            return "No se pudo eliminar la ruta o no te pertenece."
    except Exception as e:
        return f"Error de base de datos: {str(e)}"

@tool
def consultar_manual(pregunta: str):
    """
    Usa esta herramienta siempre que el usuario pregunte cómo funciona el sistema,
    sobre colores, funcionalidades, menús, o cualquier duda teórica sobre la aplicación.
    """
    if rag_chain is None:
        return "Error: El sistema de manual no está disponible (archivos de índice no encontrados)."
    
    try:
        # Se invoca toda la cadena RAG para obtener los documentos relevantes
        docs = rag_chain.invoke(pregunta)
        
        # Formateamos los documentos recuperados para que el LLM del agente los lea
        contexto_texto = "\n\n".join([
            f"Sección del Manual (Fuente: {d.metadata.get('source')}) ---\n{d.page_content}" 
            for d in docs
        ])
        
        if not contexto_texto:
            return "No encontré información relevante en el manual sobre ese tema específico."
            
        return contexto_texto
        
    except Exception as e:
        return f"Error al consultar el manual: {str(e)}"

# DEFINICIÓN DEL ESTADO Y GRAFO

class AgentState(TypedDict):
    messages: List[Annotated[str, "Mensajes del chat"]]
    contexto_visual: dict
    user_id: int

def chatbot_node(state: AgentState):
    llm = ChatOpenAI(model="gpt-5-mini", reasoning={ "effort": "low" }, 
                     text={ "verbosity": "low" }, openai_api_key=OPENAI_API_KEY)
    
    tools = [eliminar_ruta_por_posicion, consultar_manual]
    llm_with_tools = llm.bind_tools(tools)
    
    contexto = state.get('contexto_visual', {})
    
    # Prompt del sistema reforzado
    sys_msg = f"""Eres el asistente inteligente de LogistiClima.

    CONTEXTO VISUAL DEL USUARIO:
    - Pagina actual: {contexto.get('pagina', 'Desconocida')}
    - Datos visibles: {json.dumps(contexto.get('datos', {}), ensure_ascii=False)}
    - IDs técnicos (ocultos): {contexto.get('ids_rutas', [])}

    INSTRUCCIONES:
    1. Si el usuario pide una acción sobre los datos (ej: "borra la ruta 2"), usa las herramientas de acción (eliminar_ruta...).
    2. Si el usuario tiene dudas sobre cómo usar la app o qué significan las cosas (ej: "qué es el color rojo", "cómo guardo"), USA 'consultar_manual'. 
       NO inventes respuestas, busca en el manual primero.
    3. Responde de forma amable y concisa.
    """
    
    # Manejo de historial: concatenamos SystemMessage + Historial previo
    messages = [SystemMessage(content=sys_msg)] + state['messages']
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Construcción del grafo
graph_builder = StateGraph(AgentState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", ToolNode([eliminar_ruta_por_posicion, consultar_manual]))

graph_builder.set_entry_point("chatbot")

def route_tools(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges("chatbot", route_tools)
graph_builder.add_edge("tools", "chatbot")

app_ia = graph_builder.compile()

# Función pública para llamar desde Flask
def procesar_chat(mensaje_usuario, contexto_visual, historial, user_id):
    inputs = {
        "messages": [HumanMessage(content=mensaje_usuario)],
        "contexto_visual": contexto_visual,
        "user_id": user_id
    }
    
    # Ejecutamos el grafo
    resultado = app_ia.invoke(inputs)
    
    # Obtenemos la última respuesta del asistente
    ultimo_mensaje = resultado['messages'][-1]
    return ultimo_mensaje.content