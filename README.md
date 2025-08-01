# agentRAG

A local AI-powered question-answering system that uses Retrieval-Augmented
Generation (RAG) to answer questions about a pizza restaurant based on customer
reviews. The system combines LangChain, Docker-hosted models, and ChromaDB to
create an intelligent chatbot that can provide contextual answers by retrieving
relevant restaurant reviews.

## Overview

This project demonstrates a complete RAG implementation that:

- Loads restaurant review data from CSV files and other document formats
- Creates vector embeddings using Docker-hosted embedding models
- Stores embeddings in a ChromaDB vector database
- Retrieves relevant reviews based on user questions
- Uses a Docker-hosted LLM to generate contextual responses

## Features

- **Docker-Hosted AI Models**: Uses Docker model runner for hosting chat and
  embedding models
- **Multi-Format Document Support**: Supports CSV, Markdown, HTML, Python,
  and YAML files
- **Vector Search**: Semantic search through restaurant reviews using embeddings
- **Persistent Storage**: ChromaDB vector store with persistent storage
- **Interactive Chat**: Command-line interface for asking questions
- **Context-Aware Responses**: Provides answers based on relevant review context
- **Environment Configuration**: Flexible configuration via environment variables

## Prerequisites

- Python 3.8+
- Docker with model runner service hosting:
  - Chat completion API (llama.cpp or LLM Studio engine)
  - Embeddings API (llama.cpp or LLM Studio engine)
- Docker model runner running on accessible host

### Model Runner Setup

The system supports two different model runner configurations:

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

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd agentRAG
   ```

2. **Set up virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:

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

5. **Prepare data files**:
   Place your data files in the appropriate subdirectories under `./data/`:
   - CSV files in `./data/csv/`
   - Markdown files in `./data/markdown/`
   - HTML files in `./data/html/`
   - Python files in `./data/py/`
   - YAML files in `./data/yaml/`

## Project Structure

```text
agentRAG/
├── main.py                           # Main application entry point
├── vector.py                         # Vector store setup and retriever
├── requirements.txt                  # Python dependencies
├── dmodelrun.sh                     # Docker model runner test script
├── .env                             # Environment configuration
├── README.md                        # Project documentation
├── data/                            # Data directory
│   ├── csv/                         # CSV files (e.g., restaurant reviews)
│   ├── markdown/                    # Markdown documents
│   ├── html/                        # HTML documents
│   ├── py/                          # Python files
│   └── yaml/                        # YAML files
└── chrome_langchain_db/             # ChromaDB storage (created on first run)
```

## Usage

1. **Start the Docker model runner** (ensure it's accessible at the configured host)

2. **Test the model runner** (optional):

   ```bash
   ./dmodelrun.sh
   ```

3. **Start the application**:

   ```bash
   python main.py
   ```

4. **Ask questions about the restaurant**:

   ```text
   Ask your question (q to quit): What do customers think about the pizza quality?
   ```

5. **Exit the application**:
   Type `q` and press Enter to quit.

## How It Works

### Data Processing (`vector.py`)

1. **Environment Configuration**: Loads configuration from `.env` file
2. **Multi-Format Data Loading**: Reads documents from various formats:
   - **CSV**: Restaurant reviews with title, review, rating, and date
   - **Markdown**: Documentation and text files
   - **HTML**: Web content and structured documents
   - **Python**: Code files and scripts
   - **YAML**: Configuration and structured data
3. **Document Creation**: Converts each document into LangChain Documents with
   appropriate metadata
4. **Embedding Generation**: Uses Docker-hosted embedding model via HTTP API
5. **Vector Storage**: Stores embeddings in ChromaDB with persistent storage
6. **Retriever Setup**: Configures retriever to return top 5 most relevant documents

### Question Answering (`main.py`)

1. **Environment Setup**: Loads LLM and system configuration from environment
   variables
2. **User Input**: Accepts questions through command-line interface
3. **Context Retrieval**: Uses the retriever to find relevant documents based on
   the question
4. **Prompt Construction**: Combines retrieved documents with the user question
   in a structured prompt
5. **Response Generation**: Uses Docker-hosted LLM via HTTP API to generate
   contextual answers
6. **Output**: Displays the AI-generated response to the user

## Technical Details

### Dependencies

- **python-dotenv**: Environment variable management
- **pandas**: Data manipulation and CSV processing
- **requests**: HTTP client for Docker model runner communication
- **chromadb**: Vector database for embeddings storage
- **langchain-core**: Core LangChain framework components
- **langchain-chroma**: ChromaDB integration for LangChain
- **langchain-community**: Community document loaders
- **langchain-ollama**: Ollama integration (retained for compatibility)

### Architecture

- **Custom HTTP Clients**: Direct HTTP communication with Docker model runner
- **LocalLLM Class**: Custom LangChain Runnable for Docker-hosted chat completions
- **LocalEmbeddings Class**: Custom embedding client for Docker-hosted embeddings
- **Multi-Format Support**: Extensible document loading system

### Vector Database

- **Database**: ChromaDB with persistent storage
- **Collection**: Configurable via `COLLECTION_NAME` environment variable
- **Storage Location**: Configurable via `DB_LOCATION` environment variable
- **Search Configuration**: Returns top 5 most similar documents (k=5)

## Configuration

### Environment Variables

#### Chat Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_HOST` | `LLM_STUDIO` | Chat host selection (`LLAMA_CPP` or `LLM_STUDIO`) |
| `LLM_HOST_LLAMA_CPP` | `http://localhost:12434` | llama.cpp host URL |
| `LLM_PATH_LLAMA_CPP` | `engines/llama.cpp/v1/chat/completions` | llama.cpp chat API path |
| `OLLAMA_MODEL_LLAMA_CPP` | `ai/llama3.2:latest` | llama.cpp chat model |
| `LLM_HOST_LLM_STUDIO` | `http://localhost:1234` | LLM Studio host URL |
| `LLM_PATH_LLM_STUDIO` | `v1/chat/completions` | LLM Studio chat API path |
| `OLLAMA_MODEL_LLM_STUDIO` | `liquid/lfm2-1.2b` | LLM Studio chat model |

#### Embedding Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_HOST` | `LLM_STUDIO` | Embedding host selection (`LLAMA_CPP` or `LLM_STUDIO`) |
| `EMBEDDING_HOST_LLAMA_CPP` | `http://localhost:12434` | llama.cpp embedding host URL |
| `EMBEDDING_PATH_LLAMA_CPP` | `llama.cpp/v1/embeddings` | llama.cpp embedding API path |
| `EMBEDDING_MODEL_LLAMA_CPP` | `ai/mxbai-embed-large` | llama.cpp embedding model |
| `EMBEDDING_DIMENSION_LLAMA_CPP` | `1024` | llama.cpp embedding dimension |
| `EMBEDDING_HOST_LLM_STUDIO` | `http://localhost:1234` | LLM Studio embedding host URL |
| `EMBEDDING_PATH_LLM_STUDIO` | `v1/embeddings` | LLM Studio embedding API path |
| `EMBEDDING_MODEL_LLM_STUDIO` | `nomic-embed-text-v1.5` | LLM Studio embedding model |
| `EMBEDDING_DIMENSION_LLM_STUDIO` | `768` | LLM Studio embedding dimension |

#### System Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `DB_LOCATION` | `./faiss_index` | ChromaDB storage directory |
| `COLLECTION_NAME` | `PizzaBoys` | ChromaDB collection name |
| `DATA_FOLDER` | `./data` | Data directory path |
| `ROLE_SYSTEM_CONTENT` | `You are a helpful assistant.` | System message |
| `REBUILD_VECTOR` | `false` | Force rebuild vector store on startup |
| `CHUNK_SIZE` | `500` | Document chunk size |
| `CHUNK_OVERLAP` | `100` | Document chunk overlap |
| `RETRIEVAL_TOP_K` | `5` | Number of documents to retrieve |

### Customization

#### Changing the Domain

To adapt this system for a different domain:

1. **Update the data source**: Place your data files in the appropriate
   `./data/` subdirectories
2. **Modify the prompt**: Update the `PROMPT_TEMPLATE` environment variable or
   modify the default in `main.py`:

   ```python
   PROMPT_TEMPLATE_STR = os.getenv(
       "PROMPT_TEMPLATE",
       "You are an expert in answering questions about [YOUR DOMAIN].\n\n"
       "Here are some relevant documents: {reviews}\n\n"
       "Here is the question to answer: {question}"
   )
   ```

#### Adjusting Retrieval

Modify the retriever configuration in `vector.py`:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Retrieve more documents
```

#### Using Different Models

Update the environment variables based on your chosen engine:

**For llama.cpp engine:**
```bash
LLM_HOST=LLAMA_CPP
OLLAMA_MODEL_LLAMA_CPP=ai/llama3.1:latest
EMBEDDING_MODEL_LLAMA_CPP=ai/nomic-embed-text:latest
```

**For LLM Studio engine:**
```bash
LLM_HOST=LLM_STUDIO
OLLAMA_MODEL_LLM_STUDIO=liquid/lfm2-1.5b
EMBEDDING_MODEL_LLM_STUDIO=nomic-embed-text-v2:latest
```

## Troubleshooting

### Common Issues

1. **Docker model runner not accessible**:
   - Verify the Docker service is running
   - Check the host configuration in `.env`
   - Test with `./dmodelrun.sh`

2. **Model not found errors**:
   - Ensure the specified models are available in your Docker model runner
   - Check model names match exactly

3. **Data files not found**:
   - Verify data files exist in the correct `./data/` subdirectories
   - Check file permissions

4. **ChromaDB permission errors**:
   - Check write permissions for the `DB_LOCATION` directory
   - Ensure the directory is writable by the application

5. **Environment variable issues**:
   - Verify `.env` file exists and is readable
   - Check environment variable names and values

### Performance Considerations

- **First run**: Initial setup takes longer as it processes all documents and
  creates embeddings
- **Subsequent runs**: Faster startup as vector database is persisted (unless
  `REBUILD_VECTOR=true`)
- **Memory usage**: Depends on the size of your dataset and Docker model runner configuration
- **Network latency**: Performance depends on Docker model runner response times

### Development and Testing

Use the included test script to verify Docker model runner connectivity:

```bash
chmod +x dmodelrun.sh
./dmodelrun.sh
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with your Docker model runner setup
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Docker-based model serving for scalable AI deployment
