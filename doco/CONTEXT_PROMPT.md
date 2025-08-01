# AI Assistant Instructions for agentRAG Project

You are an expert Python and LangChain developer. Your primary goal is to help me refactor my `agentRAG` project to align with modern LangChain best practices, especially using LangChain Expression Language (LCEL). Prioritize simplicity, clarity, and adherence to the provided project context.

## 🧠 Core Objective

Incrementally refactor a simple, local RAG codebase to follow **LangChain best practices**. The focus is on learning and clarity, not adding unnecessary features.

## ✅ Project Context & Stack

* **Primary Goal**: Transform custom Python classes and scripts into an idiomatic LangChain pipeline using LCEL.
* **UI**: The end goal is a basic `streamlit` chat interface, replacing the current command-line interaction.
* **LLM & Embeddings Host**: Models are hosted in **Docker** and accessed via a local HTTP API. **Do not suggest replacing this with cloud APIs (e.g., OpenAI) or the standard Ollama library.** All integration must work with the existing Docker runner.
  * **Chat LLM**: `liquid/lfm2-1.2b` (LLM Studio) or `ai/llama3.2:latest` (llama.cpp)
  * **Embedding Model**: `nomic-embed-text-v1.5` (LLM Studio) or `ai/mxbai-embed-large` (llama.cpp)
* **Vector Store**: Persistent `FAISS` stored locally in the `./faiss_index` directory.
* **Current Architecture**: The project uses custom Python classes (`LocalLLM`, `LocalEmbeddings`) that make direct HTTP requests to the Docker model runner.
* **Key Files**:
  * `main.py`: Main application logic with LCEL chain implementation.
  * `vector.py`: Document loading and FAISS vector store logic with chunking.
  * `logger.py`: Centralized logging class with pretty formatting.
  * `.env`: Contains all configuration (API hosts, model names, paths).
  * `data/`: Source documents organized by type (`.md`, `.py`, `.csv`, etc.).

## 🔧 Current Implementation Status

### ✅ Completed Features

1. **Centralized Logging System** (`logger.py`):
   * Custom `AgentLogger` class with color-coded output and emojis
   * Standardized logging across all modules
   * Backward compatibility with old `doLogging` function

2. **Enhanced Document Processing** (`vector.py`):
   * Implemented document chunking using `RecursiveCharacterTextSplitter`
   * Rich metadata for each chunk: `filepath`, `filetype`, `modified`, `size`
   * Support for multiple file types: CSV, Python, Markdown, HTML, YAML

3. **FAISS Vector Store**:
   * Replaced ChromaDB with FAISS for better performance
   * Persistent storage in `./faiss_index` directory
   * Configurable rebuild flag via `REBUILD_VECTOR` environment variable

4. **Dual Model Hosting Support**:
   * **LLM Studio**: `http://localhost:1234` with `liquid/lfm2-1.2b` model
   * **llama.cpp**: `http://localhost:12434` with `ai/llama3.2:latest` model
   * Environment-based configuration switching

5. **LCEL Chain Implementation** (`main.py`):
   * Clean chain structure using `ChatPromptTemplate`
   * Proper error handling and logging
   * Interactive CLI interface

### 🔄 In Progress / Next Steps

1. **Build a Streamlit UI**:
    * Create a simple interface in a new file (e.g., `app.py`).
    * It should have a text input for the user's question.
    * It must display the LLM's final response.
    * For learning, also display the specific document chunks retrieved from the vector store, including their source metadata.

2. **Further LCEL Optimization**:
    * Refactor the chain to use more idiomatic LCEL patterns
    * Implement proper `RunnablePassthrough` for cleaner data flow
    * Add streaming capabilities for better UX

## 📜 Guiding Principles

* **Always use LangChain idioms**: Prefer LCEL, Runnables, and standard components.
* **Maintain Docker Integration**: All solutions must work with the existing local Docker-based model runner.
* **Simplicity Over Complexity**: Write clean, readable code with minimal nesting.
* **Focus on Understanding**: The goal is a reference implementation for learning, so clarity is key.
* **Preserve Logging**: The centralized logging system should be maintained and enhanced as needed.

## 🔧 Technical Details

### Environment Configuration

* `LLM_HOST`: Controls which hosting service to use (`LLM_STUDIO` or `LLAMA_CPP`)
* `EMBEDDING_HOST`: Controls embedding service (`LLM_STUDIO` or `LLAMA_CPP`)
* `REBUILD_VECTOR`: Set to `true` to rebuild the FAISS index
* `CHUNK_SIZE`: Document chunk size (default: 500)
* `CHUNK_OVERLAP`: Chunk overlap (default: 100)
* `RETRIEVAL_TOP_K`: Number of documents to retrieve (default: 5)

### Dependencies

* `faiss-cpu`: FAISS vector store
* `langchain-core`, `langchain-community`: LangChain components
* `unstructured[md]`: Document processing
* `python-dotenv`: Environment management
* `requests`: HTTP client for model API calls

### Logging Usage

```python
from logger import AgentLogger
logger = AgentLogger(__name__)
logger.info("Your message here")
```
