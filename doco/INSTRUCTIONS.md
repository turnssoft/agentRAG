# agentRAG Learning Instructions

This project is a *learning-focused* RAG implementation using LangChain,
Docker-hosted models, and ChromaDB. The goal is to incrementally transform this
codebase to follow **LangChain best practices** for a simple, local, RAG-based
LLM system — without adding unnecessary complexity.

---

## 🧠 Learning Objective

Adapt this codebase to follow best practices from the
[LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/) and
other official docs. Prioritize:

- Simplicity
- Clarity
- Metadata inclusion
- Best practices for document formatting

---

## ✅ Project Endstate

A **simple local RAG pipeline** with:

1. 🧑‍💻 A basic `streamlit` chat UI.
2. ✅ Best practices for formatting and embedding:
   - Markdown (`.md`)
   - HTML (`.html`)
   - Python (`.py`)
   - Ansible YAML (`.yml`, `.yaml`)
   - CSV (`.csv`)
3. 🧠 Embeddings enriched with metadata.
4. 🔗 Docker-hosted local LLM (`ai/llama3.2:latest`)
5. 🔍 Vector store with persistent ChromaDB using:
   - `ai/mxbai-embed-large:latest` embedding model

---

## 🧱 Current Environment

- **IDE**: Cursor
- **Python**: 3.10
- **Venv**: `source ~/git/agentRAG/venv/bin/activate`
- **OS**: WSL Ubuntu
- **Model Host**: Docker model runner (HTTP API)
- **Directory Structure**:

  ```text
  .
  ├── main.py
  ├── vector.py
  ├── README.md
  ├── requirements.txt
  ├── dmodelrun.sh
  ├── .env
  ├── chrome_langchain_db/
  ├── data/
  │   ├── csv/
  │   ├── markdown/
  │   ├── html/
  │   ├── py/
  │   └── yaml/
  └── doco/  # Target folder for docs to embed
  ```

---

## 🔧 Current Architecture

### Model Runner Integration

The project supports two different model runner configurations:

#### Option 1: llama.cpp Engine
- **Host**: `http://localhost:12434`
- **Chat Completions API**: `/engines/llama.cpp/v1/chat/completions`
- **Embeddings API**: `/engines/llama.cpp/v1/embeddings`
- **Default Chat Model**: `ai/llama3.2:latest`
- **Default Embedding Model**: `ai/mxbai-embed-large:latest`

#### Option 2: LLM Studio Engine
- **Host**: `http://localhost:1234`
- **Chat Completions API**: `/v1/chat/completions`
- **Embeddings API**: `/v1/embeddings`
- **Default Chat Model**: `liquid/lfm2-1.2b`
- **Default Embedding Model**: `nomic-embed-text-v1.5`

### Configuration Features

- **Flexible Host Selection**: Choose between `LLAMA_CPP` or `LLM_STUDIO` via environment variables
- **Custom HTTP Clients**: Direct HTTP communication with selected model runner
- **LocalLLM Class**: Custom LangChain Runnable for Docker-hosted chat completions
- **LocalEmbeddings Class**: Custom embedding client for Docker-hosted embeddings
- **Environment Configuration**: Flexible configuration via `.env` file

---

## 🔧 Refactoring Tasks (in order)

### 1. Replace CLI with Streamlit

- Create a basic Streamlit interface to:
  - Accept user queries
  - Display the response
  - Show retrieved chunks (for learning transparency)
  - Display source metadata

---

### 2. Enhanced Document Processing

#### Current Document Processing Implementation

The system already supports multiple file formats through the structured data folder approach:

```python
# Current loaders in vector.py
def load_csv(path: str) -> list[Document]:
    # CSV processing with metadata
    
def load_markdown(path: str) -> list[Document]:
    # Markdown processing with UnstructuredMarkdownLoader
    
def load_html(path: str) -> list[Document]:
    # HTML processing with UnstructuredHTMLLoader
    
def load_python(path: str) -> list[Document]:
    # Python file processing with PythonLoader
    
def load_yaml(path: str) -> list[Document]:
    # YAML processing with TextLoader
```

#### Next Steps

- Add document chunking with RecursiveCharacterTextSplitter
- Enhance metadata extraction
- Add file modification tracking

---

### 3. Document Chunking and Metadata Enhancement

- Implement chunking:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
```

- Enhanced metadata:
  - filepath
  - filetype
  - line_start / line_end (if chunked by line)
  - source (name or tag)
  - modification_date
  - file_size

---

### 4. Embedding Strategy Migration

#### Current Embedding Implementation

```python
# Custom Docker-hosted embeddings
class LocalEmbeddings:
    def __init__(self, host: str, engine: str, model: str):
        self.url = f"{host}/engines/{engine}/v1/embeddings"
        self.model = model
```

#### Target Embedding Implementation

Migrate to LangChain-compatible embeddings while maintaining Docker integration:

```python
from langchain_community.embeddings import OllamaEmbeddings
# Or create a custom LangChain-compatible wrapper for Docker embeddings
```

---

### 5. Prompt and Chain Enhancement

#### Current LLM Implementation

```python
# Custom LocalLLM class with direct HTTP calls
class LocalLLM(Runnable):
    def invoke(self, input):
        # Direct HTTP API calls to Docker model runner
```

#### Target LLM Implementation

- Enhanced prompt templates:

```python
from langchain_core.prompts import ChatPromptTemplate

template = """
You are a helpful assistant with access to relevant context.

Context:
{context}

Question:
{question}

Please provide a helpful and accurate response based on the context provided.
"""

prompt = ChatPromptTemplate.from_template(template)
```

- Improved chain structure:

```python
from langchain_core.runnables import RunnableMap
chain = prompt | llm
```

---

## 🧩 Optional Enhancements

- Add document preview per chunk in Streamlit
- Filter documents by source file
- Add metadata tags to Streamlit sidebar
- Implement document similarity scoring
- Add conversation history
- Real-time document monitoring and re-indexing

---

## 🧼 Style Notes

- Only comment where clarity is needed
- Avoid excess nesting
- Stick to LangChain idioms
- Minimize options, maximize understanding
- Maintain Docker integration flexibility

---

## 🔗 References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain ChromaDB Integration](https://python.langchain.com/docs/integrations/vectorstores/chroma/)
- [LangChain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)
- [LangChain Runnables](https://python.langchain.com/docs/concepts/runnables/)

---

## 🚀 Environment Setup

### Prerequisites

1. Docker model runner service running
2. Python 3.10+ with venv
3. Required environment variables in `.env`:

```bash
# ────────────────────────────────────────────────────────────────────────────
# -- Chat Settings ──────────────────────────────────────────────────────────

# ─── Chat host ─────────────────────────────────────────────────────────────
LLM_HOST=LLM_STUDIO  # Options: LLAMA_CPP or LLM_STUDIO

# ─── Chat host llama.cpp ────────────────────────────────────────────────────
LLM_HOST_LLAMA_CPP=http://localhost:12434
LLM_PATH_LLAMA_CPP=engines/llama.cpp/v1/chat/completions
OLLAMA_MODEL_LLAMA_CPP=ai/llama3.2:latest

# ─── Chat host LLM Studio ────────────────────────────────────────────────────
LLM_HOST_LLM_STUDIO=http://localhost:1234
LLM_PATH_LLM_STUDIO=v1/chat/completions
OLLAMA_MODEL_LLM_STUDIO=liquid/lfm2-1.2b

# ─── System & Prompt ────────────────────────────────────────────────────────
ROLE_SYSTEM_CONTENT=You are a helpful assistant.

# ────────────────────────────────────────────────────────────────────────────
# -- Embedding Settings ──────────────────────────────────────────────────────

# ─── Embedding host ────────────────────────────────────────────────────────
EMBEDDING_HOST=LLM_STUDIO  # Options: LLAMA_CPP or LLM_STUDIO

# ─── Embedding host llama.cpp ─────────────────────────────────────────────
EMBEDDING_HOST_LLAMA_CPP=http://localhost:12434
EMBEDDING_PATH_LLAMA_CPP=llama.cpp/v1/embeddings
EMBEDDING_MODEL_LLAMA_CPP=ai/mxbai-embed-large
EMBEDDING_DIMENSION_LLAMA_CPP=1024

# ─── Embedding host LLM Studio ─────────────────────────────────────────────
EMBEDDING_HOST_LLM_STUDIO=http://localhost:1234
EMBEDDING_PATH_LLM_STUDIO=v1/embeddings
EMBEDDING_MODEL_LLM_STUDIO=nomic-embed-text-v1.5
EMBEDDING_DIMENSION_LLM_STUDIO=768

# ─── Vector store rebuild toggle ───────────────────────────────────────────
REBUILD_VECTOR=false

# ─── ChromaDB backend engine ───────────────────────────────────────────────
DB_LOCATION="./faiss_index"
COLLECTION_NAME="PizzaBoys"

# ─── Data folder ────────────────────────────────────────────────────────────
DATA_FOLDER="./data"

# ─── Splitter settings ──────────────────────────────────────────────────────
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# ─── Top K for retrieval ───────────────────────────────────────────────────
RETRIEVAL_TOP_K=5
```

### Testing

Use the included test script to verify model runner connectivity:

```bash
chmod +x dmodelrun.sh
./dmodelrun.sh
```

### Switching Between Model Runners

To switch between llama.cpp and LLM Studio engines:

1. **For llama.cpp engine:**
   ```bash
   LLM_HOST=LLAMA_CPP
   EMBEDDING_HOST=LLAMA_CPP
   ```

2. **For LLM Studio engine:**
   ```bash
   LLM_HOST=LLM_STUDIO
   EMBEDDING_HOST=LLM_STUDIO
   ```

The system will automatically use the appropriate host URLs, API paths, and models based on your selection.

---
