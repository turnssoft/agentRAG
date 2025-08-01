# Standard library imports
import datetime
import os
import shutil
import pandas as pd
from pathlib import Path
import yaml

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

# Log configuration after environment variables are loaded
logger.info("Vector module configuration loaded:")
logger.info(f"  Data folder: {DATA_FOLDER}")
logger.info(f"  Database location: {DB_LOCATION}")
logger.info(f"  Rebuild flag: {REBUILD}")
logger.info(f"  Chunk size: {CHUNK_SIZE}")
logger.info(f"  Chunk overlap: {CHUNK_OVERLAP}")
logger.info(f"  Top K retrieval: {TOP_K}")
logger.info(f"  Embedding service: {'LLM Studio' if IS_STUDIO else 'Llama.cpp'}")
logger.info(f"  Embedding host: {EMBED_HOST}")
logger.info(f"  Embedding model: {EMBED_MODEL}")
logger.info(f"  Embedding dimension: {EMBED_DIM}")

def load_csv(path: str) -> list[Document]:
    """
    Loads data from a CSV file, dynamically identifying the main content column
    and treating all other columns as metadata.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        list[Document]: A list of Document objects, one for each row in the CSV.
    """
    logger.info(f"Dynamically loading CSV file with pandas: {path}")
    try:
        df = pd.read_csv(path)
        
        # Convert all columns to string type to handle mixed data gracefully
        df = df.astype(str)

        # Find the column with the most text content to use as page_content
        # This is a heuristic that assumes the longest field is the main content.
        content_column = df.apply(lambda x: x.str.len()).sum().idxmax()
        logger.debug(f"Identified '{content_column}' as the primary content column for {path}")

        docs = []
        for index, row in df.iterrows():
            # Use the identified column for the main page content
            page_content = row[content_column]
            
            # Use all *other* columns as metadata
            metadata = {
                "source": path,
                "row": index
            }
            # Add all columns other than the content column to metadata
            for col in df.columns:
                if col != content_column:
                    metadata[col.lower()] = row[col]
            
            doc = Document(page_content=page_content, metadata=metadata)
            docs.append(doc)
            
        logger.info(f"✓ Created {len(docs)} documents from {path}")
        if docs:
            logger.debug(f"Metadata of first document in {path}: {docs[0].metadata}")
            logger.debug(f"Content of first document in {path}: {docs[0].page_content}")
        return docs

    except Exception as e:
        logger.error(f"Pandas CSV loader failed for {path} ({e}), falling back to TextLoader")
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()

def load_python(path: str) -> list[Document]:
    logger.info(f"Loading Python file: {path}")
    try:
        loader = PythonLoader(path)
        raw_docs = loader.load()
        file = Path(path)
        file_name = file.stem
        file_ext = file.suffix
        file_date = datetime.datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d")
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(raw_docs)
        for chunk in chunks:
            chunk.metadata.update({
                "source": f"{file_name}{file_ext}",
                "date": file_date,
                "type": "python",
                "file_name": file_name,
                "file_ext": file_ext,
                "file_date": file_date
            })
        return chunks
    except Exception as e:
        logger.error(f"Python loader failed ({e}), falling back to TextLoader with chunking")
        loader = TextLoader(path)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)

def load_yaml(path: str) -> list[Document]:
    logger.info(f"Loading YAML file: {path}")
    try:
        loader = TextLoader(path, encoding="utf-8")
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)
    except Exception as e:
        logger.error(f"YAML loader failed ({e}), falling back to TextLoader")
        loader = TextLoader(path)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)

def load_ansible_yaml(path: str) -> list[Document]:
    logger.info(f"Parsing Ansible YAML: {path}")
    documents = []
    file = Path(path)
    stats = file.stat()

    with open(path, 'r', encoding='utf-8') as f:
        try:
            plays = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error: {e}")
            return []

    for play in plays if isinstance(plays, list) else [plays]:
        tasks = play.get("tasks", [])
        for i, task in enumerate(tasks):
            name = task.get("name", f"unnamed_task_{i}")
            doc = Document(
                page_content=yaml.dump(task, sort_keys=False),
                metadata={
                    "filepath": str(file),
                    "filetype": "yaml",
                    "modification_time": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "file_size": stats.st_size,
                    "task_name": name,
                    "task_index": i
                }
            )
            documents.append(doc)
    return documents

def load_markdown(path: str) -> list[Document]:
    logger.info(f"Loading markdown file: {path}")
    try:
        loader = UnstructuredMarkdownLoader(path, mode="elements")
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)
    except Exception as e:
        logger.error(f"Markdown loader failed ({e}), falling back to TextLoader")
        loader = TextLoader(path, encoding="utf-8")
        return loader.load_and_split()

def load_html(path: str) -> list[Document]:
    logger.info(f"Loading HTML file: {path}")
    try:
        raw_docs = UnstructuredHTMLLoader(path, encoding="utf-8", mode="elements").load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)
    except Exception as e:
        logger.error(f"HTML loader failed ({e}), falling back to TextLoader")
        loader = TextLoader(path, encoding="utf-8")
        return loader.load_and_split()


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
            logger.debug(f"Generated embedding of dimension {len(embedding)} for text of length {len(text)}")
            return embedding
        except requests.RequestException as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = [self._embed(t) for t in texts]
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query string."""
        logger.debug(f"Generating query embedding for text of length {len(text)}")
        return self._embed(text)

# ===== VECTOR STORE FUNCTIONS =====
def get_vector_store() -> FAISS:
    """Load existing FAISS vector store or create a new empty one."""
    store_dir = str(DB_LOCATION)
    index_file = DB_LOCATION / "index.faiss"
    
    if index_file.exists():
        logger.info(f"Loading existing FAISS index from {store_dir}")
        try:
            store = FAISS.load_local(
                store_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("✓ Existing FAISS index loaded successfully")
            return store
        except Exception as e:
            logger.error(f"Failed to load existing FAISS index: {e}")
            logger.info("Creating new FAISS index as fallback")
    
    # Create new empty FAISS index
    logger.info(f"Creating new FAISS index with dimension {EMBED_DIM}")
    index = faiss.IndexFlatL2(EMBED_DIM)
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    logger.info("✓ New FAISS index created successfully")
    return store

def load_documents() -> list[Document]:
    """ Load documents by subdir """
    logger.info("Starting document loading process...")
    docs = []
    total_files = 0
    successful_files = 0
    
    for subdir, loader in (
        ("csv", load_csv),
        ("python", load_python),
        ("yaml", load_yaml),
        ("ansible", load_ansible_yaml),
        ("markdown", load_markdown),
        ("html", load_html),
    ): 
        dir_path = os.path.join(DATA_FOLDER, subdir)
        if os.path.exists(dir_path):
            logger.info(f"Processing {subdir} directory: {dir_path}")
            files_in_dir = os.listdir(dir_path)
            logger.info(f"Found {len(files_in_dir)} files in {subdir} directory")
            
            for file in files_in_dir:
                file_path = os.path.join(dir_path, file)
                total_files += 1
                try:
                    file_docs = loader(file_path)
                    docs.extend(file_docs)
                    successful_files += 1
                    logger.debug(f"Successfully loaded {len(file_docs)} documents from {file}")
                except Exception as e:  
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        else:
            logger.debug(f"Directory not found: {dir_path}")
    
    logger.info(f"Document loading completed: {successful_files}/{total_files} files processed successfully")
    logger.info(f"Total documents loaded: {len(docs)}")
    return docs


def build_vector_store():
    """Build or rebuild the FAISS vector store, then return it."""
    logger.info("Building vector store...")
    
    # Handle rebuild flag - remove existing index if requested
    if REBUILD and DB_LOCATION.exists():
        logger.info(f"REBUILD flag set - removing existing index at {DB_LOCATION}")
        shutil.rmtree(DB_LOCATION)
        logger.info("✓ Existing index removed successfully")
    
    # Get or create vector store
    store = get_vector_store()
    
    # Build index if it doesn't exist
    if not DB_LOCATION.exists():
        logger.info("No existing index found - building new index")
        docs = load_documents()
        
        if docs:
            logger.info(f"Adding {len(docs)} documents to vector store")
            store.add_documents(docs)
            
            logger.info(f"Saving vector store to {DB_LOCATION}")
            store.save_local(str(DB_LOCATION))
            
            logger.info(f"✓ Successfully indexed {len(docs)} documents")
        else:
            logger.warning("⚠ No documents found to index - vector store will be empty")
    else:
        logger.info("✓ Using existing vector store index")
    
    return store


# ===== INITIALIZATION =====
# Create singleton embedding instance
logger.info("Initializing embedding service...")
embeddings = LocalEmbeddings(
    host=EMBED_HOST,
    path=EMBED_PATH,
    model=EMBED_MODEL,
    studio=IS_STUDIO,
)

# Initialize retriever for queries
logger.info("Building retriever...")
retriever = build_vector_store().as_retriever(search_kwargs={"k": TOP_K})
logger.info(f"✓ Retriever ready - will return top {TOP_K} results per query")