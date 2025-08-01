# vector.py

import os
import shutil
import yaml
import datetime
import pandas as pd
import requests
import faiss
from dotenv import load_dotenv

# ─── Load environment variables ──────────────────────────────────────────────
load_dotenv()
DB_LOCATION     = os.getenv("DB_LOCATION", "./faiss_index")  # Update if desired

# ─── Collection and data folder locations ─────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "restaurant_reviews")  # Not used in FAISS
DATA_FOLDER     = os.getenv("DATA_FOLDER", "./data")

# ─── Embedding service configuration ───────────────────────────────────────
EMBEDDING_HOST   = os.getenv("EMBEDDING_HOST")
# if LLM_STUDIO then EMBEDDING_HOST_LLM_STUDIO ELSE EMBEDDING_HOST_LLAMA_CPP
if EMBEDDING_HOST == "LLM_STUDIO":
    EMBEDDING_HOST = os.getenv("EMBEDDING_HOST_LLM_STUDIO")
    EMBEDDING_PATH = os.getenv("EMBEDDING_PATH_LLM_STUDIO")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_LLM_STUDIO")
else:
    EMBEDDING_HOST = os.getenv("EMBEDDING_HOST_LLAMA_CPP")
    EMBEDDING_PATH = os.getenv("EMBEDDING_PATH_LLAMA_CPP")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_LLAMA_CPP")


# ─── Chunk size and overlap ────────────────────────────────────────────────
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 200))

# ─── Top K for retrieval ───────────────────────────────────────────────────
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

# ─── Toggle to rebuild the vector store on each run ────────────────────────
REBUILD_FLAG     = os.getenv("REBUILD_VECTOR", "false").lower() in ("1", "true", "yes")

# ─── Prepare or rebuild local FAISS files ──────────────────────────────────
if REBUILD_FLAG and os.path.exists(DB_LOCATION):
    shutil.rmtree(DB_LOCATION, ignore_errors=True)
add_documents = not os.path.exists(DB_LOCATION)

# ─── Standard imports ──────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    PythonLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from pathlib import Path

# ─── Custom Embeddings via HTTP ────────────────────────────────────────────
class LocalEmbeddings(Embeddings):
    def __init__(self, host: str, path: str, model: str, is_llm_studio: bool = False):
        self.url = f"{host}/{path}"
        self.model = model
        self.is_llm_studio = is_llm_studio

    def _embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "input": text}
        
        # Set up headers - only add Authorization for LLM Studio
        headers = {"Content-Type": "application/json"}
        if self.is_llm_studio:
            headers["Authorization"] = "Bearer lm-studio"
            
        try:
            resp = requests.post(
                self.url,
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [{}])[0].get("embedding", [])
        except Exception as e:
            print(f"\n❌ Embedding failed.\nText (truncated): {text[:200]}...\nChars: {len(text)}\n")
            print(f"Error: {e}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

# Instantiate the embedding client
embeddings = LocalEmbeddings(
    host=EMBEDDING_HOST,
    path=EMBEDDING_PATH,
    model=EMBEDDING_MODEL,
    is_llm_studio=(os.getenv("EMBEDDING_HOST") == "LLM_STUDIO")
)

# if the embeddings are successful I want to print the embeddings variable
print(f"Embeddings: {embeddings}")

def load_csv(path: str) -> list[Document]:
    print(f"Loading CSV file: {path}")
    try:
        loader = CSVLoader(path)
        raw_docs = loader.load()
        return raw_docs
    except Exception as e:
        print(f"\n❌ CSV loader failed ({e}), falling back to TextLoader")
        loader = TextLoader(path, encoding="utf-8")
        return loader.load_and_split()

def load_python(path: str) -> list[Document]:
    print(f"Loading Python file: {path}")
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
        print(f"Python loader failed ({e}), falling back to TextLoader with chunking")
        loader = TextLoader(path)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)

def load_yaml(path: str) -> list[Document]:
    try:
        loader = TextLoader(path, encoding="utf-8")
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)
    except Exception:
        loader = TextLoader(path)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)

def load_ansible_yaml(path: str) -> list[Document]:
    print(f"Parsing Ansible YAML: {path}")
    documents = []
    file = Path(path)
    stats = file.stat()

    with open(path, 'r', encoding='utf-8') as f:
        try:
            plays = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"YAML parse error: {e}")
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
    print(f"Loading markdown file: {path}")
    try:
        loader = UnstructuredMarkdownLoader(path, mode="elements")
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)
    except Exception as e:
        print(f"\n ❌ Markdown loader failed ({e}), falling back to TextLoader")
        loader = TextLoader(path, encoding="utf-8")
        return loader.load_and_split()

def load_html(path: str) -> list[Document]:
    print(f"Loading HTML file: {path}")
    try:
        raw_docs = UnstructuredHTMLLoader(path, encoding="utf-8", mode="elements").load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return splitter.split_documents(raw_docs)
    except Exception as e:
        print(f"HTML loader failed ({e}), falling back to TextLoader")
        loader = TextLoader(path, encoding="utf-8")
        return loader.load_and_split()

# ─── Aggregate all documents from DATA_FOLDER ───────────────────────────────
documents: list[Document] = []
if add_documents:
    for subdir, loader in (
        ("csv", load_csv),
        ("py", load_python),
        ("yaml", load_yaml),
        ("ansible", load_ansible_yaml),
        ("markdown", load_markdown),
        ("html", load_html),
    ):
        dir_path = os.path.join(DATA_FOLDER, subdir)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    docs = loader(file_path)
                    documents.extend(docs)
                    print(f"Loaded {len(docs)} documents from {file_path}")
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

# ─── Initialize FAISS vector store ─────────────────────────────────────────
if os.path.exists(DB_LOCATION):
    vector_store = FAISS.load_local(DB_LOCATION, embeddings, allow_dangerous_deserialization=True)
else:
    dimension = 1024  # Dimension for mxbai-embed-large
    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore={},
        index_to_docstore_id={}
    )

if add_documents and documents:
    for i, doc in enumerate(documents):
        if not isinstance(doc, Document):
            print(f"\n❌ Skipping invalid doc #{i}: not a Document (got {type(doc)})")
            continue
        print(f"✅ Adding Document #{i}")
        print(f"  → type: {type(doc)}")
        print(f"  → content: {doc.page_content}")
        print(f"  → metadata: {doc.metadata}")
        try:
            # No need for filter_complex_metadata with FAISS
            vector_store.add_documents([doc])
        except Exception as e:
            print(f"\n❌ Failed to add document #{i}")
            print(f"  → Metadata: {doc.metadata}")
            print(f"  → Error: {e}")
    vector_store.save_local(DB_LOCATION)

# ─── Retriever ─────────────────────────────────────────────────────────────
retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})