# 🏦 Sistema Inteligente — Banco Atlas Financiero

App de Streamlit con RAG + AI Agent usando Gemini 2.5 Flash y ChromaDB.

---

## 📁 Estructura del repositorio

```
tu-repo/
├── app.py                  ← Aplicación principal
├── requirements.txt        ← Dependencias
├── README.md
└── chroma_db_atlas/        ← Base de datos vectorial (subir completa)
    ├── chroma.sqlite3
    ├── data_level0.bin
    └── length.bin
```

---

## 🚀 Deploy en Streamlit Cloud

### 1. Sube todos estos archivos a GitHub

Asegúrate de que la carpeta `chroma_db_atlas/` con sus archivos esté en el repositorio.

### 2. Configura el Secret en Streamlit Cloud

Ve a tu app → **⚙️ Settings → Secrets** y añade:

```toml
GOOGLE_API_KEY = "AIza...tu_clave_de_gemini_aqui"
```

### 3. Deploy

- Conecta tu repositorio en [share.streamlit.io](https://share.streamlit.io)
- Main file: `app.py`
- ¡Listo!

---

## 🔧 Errores comunes corregidos

| Error | Causa | Solución aplicada |
|-------|-------|-------------------|
| `TypeError` en `os.environ[...]` | `st.secrets.get()` devolvía `None` | Lectura segura con `str()` y validación |
| `ImportError: AgentExecutor` | Paquete `langchain` desactualizado | `requirements.txt` con versiones correctas |
| Ruta de Chroma `/content/drive/...` | Ruta de Colab, no existe en Streamlit | Ruta relativa con `os.path.dirname(__file__)` |
| `HuggingFaceEmbeddings` deprecated | Clase movida a `langchain-huggingface` | Importación desde `langchain_huggingface` |

---

## 🛠️ Tecnologías

- **Streamlit** — interfaz web
- **LangChain** — orquestación de LLM y agentes
- **Gemini 2.5 Flash** — modelo de lenguaje
- **ChromaDB** — base de datos vectorial
- **HuggingFace** — embeddings multilingües (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Wikipedia** — herramienta de búsqueda externa del agente
