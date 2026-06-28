import streamlit as st
import os

# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA — debe ser el primer comando de Streamlit
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Banco Atlas Inteligente",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  OBTENER API KEY DE FORMA SEGURA
# ══════════════════════════════════════════════════════════════════
api_key = None

try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = str(st.secrets["GOOGLE_API_KEY"])
except Exception:
    pass

if not api_key:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.error("❌ No se encontró **GOOGLE_API_KEY**.")
    st.info(
        "**Solución:** Ve a tu app en Streamlit Cloud → ⚙️ Settings → Secrets y añade:\n"
        "```toml\n"
        'GOOGLE_API_KEY = "AIza...tu_clave_aqui"\n'
        "```"
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ══════════════════════════════════════════════════════════════════
#  IMPORTACIONES (después de validar la key)
# ══════════════════════════════════════════════════════════════════
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.tools import tool
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
except ImportError as e:
    st.error(f"❌ Error de importación: `{e}`")
    st.info("Asegúrate de que `requirements.txt` tiene todos los paquetes necesarios.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
#  RUTA DE LA BASE DE DATOS CHROMA
# ══════════════════════════════════════════════════════════════════
# Ruta relativa al repositorio (donde está app.py)
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db_atlas")

# ══════════════════════════════════════════════════════════════════
#  CARGA DEL SISTEMA (cacheado para no recargar en cada interacción)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⏳ Cargando sistema de IA (primera vez puede tardar ~1 min)...")
def cargar_sistema():
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key,
    )
    return db, llm


try:
    db, llm = cargar_sistema()
except Exception as e:
    st.error(f"❌ Error al cargar el sistema: `{e}`")
    st.stop()

# ══════════════════════════════════════════════════════════════════
#  HERRAMIENTAS DEL AGENTE
# ══════════════════════════════════════════════════════════════════
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=800, lang="es")
wikipedia_tool_runner = WikipediaQueryRun(api_wrapper=api_wrapper)


@tool
def buscar_en_manual(consulta: str) -> str:
    """Busca políticas, procedimientos y normativas internas del Banco Atlas en su manual oficial."""
    docs = db.similarity_search(consulta, k=3)
    if not docs:
        return "No se encontraron fragmentos relevantes en el manual."
    return "\n\n---\n\n".join([d.page_content for d in docs])


@tool
def buscar_en_web(consulta: str) -> str:
    """Busca información pública y actualizada en Wikipedia sobre finanzas, economía y noticias."""
    try:
        return wikipedia_tool_runner.invoke(consulta)
    except Exception as e:
        return f"No se pudo obtener información de Wikipedia: {e}"


herramientas = [buscar_en_manual, buscar_en_web]

prompt_agente = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres el asistente inteligente oficial del Banco Atlas Financiero. "
        "Cuando te pregunten sobre políticas, procedimientos, competencias o normativas internas, "
        "usa la herramienta 'buscar_en_manual'. "
        "Cuando te pregunten sobre información externa, noticias o datos públicos, "
        "usa 'buscar_en_web'. "
        "Siempre indica claramente de qué fuente proviene tu respuesta.",
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agente = create_tool_calling_agent(llm, herramientas, prompt_agente)
ejecutor = AgentExecutor(agent=agente, tools=herramientas, verbose=False, max_iterations=5)

# ══════════════════════════════════════════════════════════════════
#  INTERFAZ DE USUARIO
# ══════════════════════════════════════════════════════════════════
st.title("🏦 Sistema Inteligente — Banco Atlas Financiero")
st.caption("Powered by Gemini 2.5 Flash · RAG con ChromaDB · AI Agent con Wikipedia")

st.divider()

# ── MODO RAG ────────────────────────────────────────────────────
st.subheader("📚 Modo RAG — Consulta el Manual Interno")
st.write("Realiza preguntas sobre las políticas, competencias y procedimientos internos del banco.")

pregunta_rag = st.text_input(
    "✏️ Escribe tu pregunta:",
    placeholder="Ej: ¿Cuáles son las competencias cardinales del banco?",
    key="input_rag",
)

if st.button("🔍 Consultar Manual", key="btn_rag"):
    if pregunta_rag.strip():
        with st.spinner("Buscando en el manual del Banco Atlas..."):
            try:
                retriever = db.as_retriever(search_kwargs={"k": 3})
                docs_recuperados = retriever.invoke(pregunta_rag)

                plantilla = (
                    "Eres un asistente corporativo del Banco Atlas Financiero.\n"
                    "Responde basándote ÚNICAMENTE en el siguiente contexto del manual.\n"
                    "Si la respuesta no está en el contexto, indícalo claramente.\n\n"
                    "Contexto:\n{context}\n\n"
                    "Pregunta: {question}\n\n"
                    "Respuesta:"
                )
                prompt = PromptTemplate.from_template(plantilla)
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                respuesta = chain.invoke(pregunta_rag)

                st.markdown("### 🤖 Respuesta")
                st.write(respuesta)

                st.markdown("### 📄 Fragmentos del manual recuperados")
                for i, doc in enumerate(docs_recuperados):
                    with st.expander(f"Fragmento {i + 1}"):
                        st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error al consultar: `{e}`")
    else:
        st.warning("⚠️ Escribe una pregunta antes de consultar.")

st.divider()

# ── MODO AGENTE ─────────────────────────────────────────────────
st.subheader("🤖 Modo Agente — Con herramientas (Manual + Wikipedia)")
st.write("El agente decide automáticamente si buscar en el manual interno o en Wikipedia.")

pregunta_agente = st.text_input(
    "✏️ Pregunta al agente:",
    placeholder="Ej: ¿Quién es el presidente del BCR del Perú?",
    key="input_agente",
)

if st.button("🚀 Consultar Agente", key="btn_agente"):
    if pregunta_agente.strip():
        with st.spinner("El agente está analizando tu pregunta y usando sus herramientas..."):
            try:
                resultado = ejecutor.invoke({"input": pregunta_agente})
                st.markdown("### ✅ Respuesta del Agente")
                st.write(resultado.get("output", "Sin respuesta."))
            except Exception as e:
                st.error(f"Error en el agente: `{e}`")
    else:
        st.warning("⚠️ Escribe una pregunta antes de consultar.")

st.divider()
st.caption("© Banco Atlas Financiero · Sistema desarrollado con LangChain + Streamlit")
