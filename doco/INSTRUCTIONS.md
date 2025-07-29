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

### Docker Model Runner Integration

The project now uses a Docker-based model runner instead of direct Ollama
integration:

- **Custom HTTP Clients**: Direct HTTP communication with Docker model runner
- **LocalLLM Class**: Custom LangChain Runnable for Docker-hosted chat completions
- **LocalEmbeddings Class**: Custom embedding client for Docker-hosted embeddings
- **Environment Configuration**: Flexible configuration via `.env` file

### API Endpoints

- **Chat Completions**: `/engines/llama.cpp/v1/chat/completions`
- **Embeddings**: `/engines/llama.cpp/v1/embeddings`

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
# Docker model runner configuration
LLM_HOST=http://localhost:12434
EMBEDDING_HOST=http://localhost:12434
DOCKER_HOST=http://localhost:12434

# Model configuration
OLLAMA_MODEL=ai/llama3.2:latest
MODEL=ai/llama3.2:latest
EMBEDDING_MODEL=ai/mxbai-embed-large:latest

# Database configuration
DB_LOCATION=./chrome_langchain_db
COLLECTION_NAME=restaurant_reviews
DATA_FOLDER=./data

# System configuration
ROLE_SYSTEM_CONTENT=You are a helpful assistant.
REBUILD_VECTOR=false
```

### Testing

Use the included test script to verify Docker model runner connectivity:

```bash
chmod +x dmodelrun.sh
./dmodelrun.sh
```

---
