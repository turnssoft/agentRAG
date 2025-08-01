# Standard library imports
import os
import shutil
from pathlib import Path

# Third-party imports
import faiss
import requests
from dotenv import load_dotenv

# LangChain imports
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import (
    CSVLoader,
    PythonLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# ===== LOGGING CONFIGURATION =====
# Import the centralized logging class
from logger import AgentLogger

# Create logger instance for this module
logger = AgentLogger(__name__)

# ===== DATA AND STORAGE CONFIGURATION =====
DATA_FOLDER = Path(os.getenv("DATA_FOLDER", "./data"))
DB_LOCATION = Path(os.getenv("DB_LOCATION", "./faiss_index"))
REBUILD = os.getenv("REBUILD_VECTOR", "false").lower() in ("1", "true", "yes")

# ===== DOCUMENT PROCESSING CONFIGURATION =====
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# ===== EMBEDDING SERVICE CONFIGURATION =====
IS_STUDIO = os.getenv("EMBEDDING_HOST", "") == "LLM_STUDIO"
EMBED_HOST = os.getenv(
    "EMBEDDING_HOST_LLM_STUDIO" if IS_STUDIO else "EMBEDDING_HOST_LLAMA_CPP"
)
EMBED_PATH = os.getenv(
    "EMBEDDING_PATH_LLM_STUDIO" if IS_STUDIO else "EMBEDDING_PATH_LLAMA_CPP"
)
EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL_LLM_STUDIO" if IS_STUDIO else "EMBEDDING_MODEL_LLAMA_CPP"
)
EMBED_DIM = int(os.getenv(
    "EMBEDDING_DIMENSION_LLM_STUDIO" if IS_STUDIO else "EMBEDDING_DIMENSION_LLAMA_CPP",
    "768" if IS_STUDIO else "1024"
))


# ===== EMBEDDING CLIENT =====
class LocalEmbeddings(Embeddings):
    """HTTP-based embedding client for Docker-hosted model runner."""

    def __init__(self, host: str, path: str, model: str, studio: bool = False):
        self.url = f"{host}/{path}"
        self.model = model
        self.studio = studio
        logger.info(f"Initialized embedding client - URL: {self.url}, Model: {self.model}, Studio: {self.studio}")

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        payload = {"model": self.model, "input": text}
        headers = {"Content-Type": "application/json"}
        
        if self.studio:
            # Add auth for LLM Studio if needed
            token = os.getenv("LLM_STUDIO_TOKEN", "")
            headers["Authorization"] = f"Bearer {token}"
        
        try:
            resp = requests.post(self.url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("data", [{}])[0].get("embedding", [])
            logger.debug( f"Generated embedding of dimension {len(embedding)} for text of length {len(text)}")
            return embedding
        except requests.RequestException as e:
            logger.error( f"Failed to generate embedding: {e}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        logger.info( f"Generating embeddings for {len(texts)} documents")
        embeddings = [self._embed(t) for t in texts]
        logger.info( f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query string."""
        logger.debug( f"Generating query embedding for text of length {len(text)}")
        return self._embed(text)

# ===== VECTOR STORE FUNCTIONS =====
def get_vector_store() -> FAISS:
    """Load existing FAISS vector store or create a new empty one."""
    store_dir = str(DB_LOCATION)
    index_file = DB_LOCATION / "index.faiss"
    
    if index_file.exists():
        logger.info( f"Loading existing FAISS index from {store_dir}")
        return FAISS.load_local(
            store_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    
    # Create new empty FAISS index
    logger.info( f"Creating new FAISS index with dimension {EMBED_DIM}")
    index = faiss.IndexFlatL2(EMBED_DIM)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

def load_documents() -> list[Document]:
    """Load and chunk all documents from DATA_FOLDER subdirectories."""
    logger.info( f"Starting document loading from {DATA_FOLDER}")
    logger.info( f"Using chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs: list[Document] = []

    # Document loader mapping by file type
    loader_map = {
        "csv": CSVLoader,
        "py": PythonLoader,
        "markdown": lambda p: UnstructuredMarkdownLoader(p, mode="elements"),
        "html": lambda p: UnstructuredHTMLLoader(p, mode="elements"),
        "yaml": lambda p: TextLoader(p, encoding="utf-8"),
        "yml": lambda p: TextLoader(p, encoding="utf-8"),
    }

    total_files_processed = 0
    total_files_skipped = 0

    for subtype, loader in loader_map.items():
        path = DATA_FOLDER / subtype
        if not path.is_dir():
            logger.debug( f"Directory {path} does not exist, skipping {subtype} files")
            continue
            
        logger.info( f"Processing {subtype} files from {path}")
        files_in_dir = list(path.iterdir())
        logger.info( f"Found {len(files_in_dir)} items in {subtype} directory")
        
        for file in files_in_dir:
            if not file.is_file():
                continue
                
            try:
                logger.debug( f"Loading file: {file.name}")
                raw_docs = loader(file).load()
                chunks = splitter.split_documents(raw_docs)
                
                # Add metadata to each chunk
                stats = file.stat()
                for chunk in chunks:
                    chunk.metadata.update({
                        "filepath": str(file),
                        "filetype": subtype,
                        "modified": stats.st_mtime,
                        "size": stats.st_size,
                    })
                
                docs.extend(chunks)
                total_files_processed += 1
                logger.info( f"✓ Loaded {len(chunks)} chunks from {file.name}")
                
            except Exception as e:
                total_files_skipped += 1
                logger.warning( f"✗ Skipping {file.name} due to error: {e}")

    logger.info( f"Document loading complete:")
    logger.info( f"  - Total documents loaded: {len(docs)}")
    logger.info( f"  - Files processed successfully: {total_files_processed}")
    logger.info( f"  - Files skipped due to errors: {total_files_skipped}")
    
    return docs


def build_vector_store():
    """Build or rebuild the FAISS vector store, then return it."""
    logger.info( "Building vector store...")
    
    # Handle rebuild flag - remove existing index if requested
    if REBUILD and DB_LOCATION.exists():
        logger.info( f"REBUILD flag set - removing existing index at {DB_LOCATION}")
        shutil.rmtree(DB_LOCATION)
    
    # Get or create vector store
    store = get_vector_store()
    
    # Build index if it doesn't exist
    if not DB_LOCATION.exists():
        logger.info( "No existing index found - building new index")
        docs = load_documents()
        
        if docs:
            logger.info( f"Adding {len(docs)} documents to vector store")
            store.add_documents(docs)
            
            logger.info( f"Saving vector store to {DB_LOCATION}")
            store.save_local(str(DB_LOCATION))
            
            logger.info( f"✓ Successfully indexed {len(docs)} documents")
        else:
            logger.warning( "⚠ No documents found to index - vector store will be empty")
    else:
        logger.info( "✓ Using existing vector store index")
    
    return store


# ===== INITIALIZATION =====
# Create singleton embedding instance
logger.info( "Initializing embedding service...")
embeddings = LocalEmbeddings(
    host=EMBED_HOST,
    path=EMBED_PATH,
    model=EMBED_MODEL,
    studio=IS_STUDIO,
)

# Initialize retriever for queries
logger.info( "Building retriever...")
retriever = build_vector_store().as_retriever(search_kwargs={"k": TOP_K})
logger.info( f"✓ Retriever ready - will return top {TOP_K} results per query")