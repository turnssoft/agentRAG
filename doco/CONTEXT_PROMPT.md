# AI Assistant Instructions for agentRAG Project

You are an expert Python and LangChain developer. Your primary goal is to help me refactor my `agentRAG` project to align with modern LangChain best practices, especially using LangChain Expression Language (LCEL). Prioritize simplicity, clarity, and adherence to the provided project context.

## 🧠 Core Objective

Incrementally refactor a simple, local RAG codebase to follow **LangChain best practices**. The focus is on learning and clarity, not adding unnecessary features.

## ✅ Project Context & Stack

* **Primary Goal**: Transform custom Python classes and scripts into an idiomatic LangChain pipeline using LCEL.
* **UI**: The end goal is a basic `streamlit` chat interface, replacing the current command-line interaction.
* **LLM & Embeddings Host**: Models are hosted in **Docker** and accessed via a local HTTP API. **Do not suggest replacing this with cloud APIs (e.g., OpenAI) or the standard Ollama library.** All integration must work with the existing Docker runner.
  * **Chat LLM**: `ai/llama3.2:latest`
  * **Embedding Model**: `ai/mxbai-embed-large:latest`
* **Vector Store**: Persistent `ChromaDB` stored locally in the `./chrome_langchain_db` directory.
* **Current Architecture**: The project uses custom Python classes (`LocalLLM`, `LocalEmbeddings`) that make direct HTTP requests to the Docker model runner.
* **Key Files**:
  * `main.py`: Main application logic.
  * `vector.py`: Document loading and vector store logic.
  * `.env`: Contains all configuration (API hosts, model names, paths).
  * `data/`: Source documents organized by type (`.md`, `.py`, `.csv`, etc.).

## 🔧 Refactoring Roadmap (In Order)

1. **Build a Streamlit UI**:
    * Create a simple interface in a new file (e.g., `app.py`).
    * It should have a text input for the user's question.
    * It must display the LLM's final response.
    * For learning, also display the specific document chunks retrieved from the vector store, including their source metadata.

2. **Enhance Document Processing (`vector.py`)**:
    * Implement document chunking using `langchain.text_splitter.RecursiveCharacterTextSplitter` (`chunk_size=800`, `chunk_overlap=200`).
    * Enrich document metadata for each chunk. Ensure metadata includes `filepath`, `filetype`, `modification_date`, and `file_size`.

3. **Migrate to LangChain-compatible Wrappers**:
    * Refactor the custom `LocalLLM` and `LocalEmbeddings` classes.
    * The goal is to make them LangChain-compatible `Runnables` or use a suitable `langchain_community` class that can be configured with a custom `base_url` for our Docker service.

4. **Improve the RAG Chain with LCEL**:
    * Create a `ChatPromptTemplate` to structure the context and question for the LLM.
    * Rebuild the core logic as a single, clean chain using LCEL, like:

```python
# Conceptual Example
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

## 📜 Guiding Principles

* **Always use LangChain idioms**: Prefer LCEL, Runnables, and standard components.
* **Maintain Docker Integration**: All solutions must work with the existing local Docker-based model runner.
* **Simplicity Over Complexity**: Write clean, readable code with minimal nesting.
* **Focus on Understanding**: The goal is a reference implementation for learning, so clarity is key.
