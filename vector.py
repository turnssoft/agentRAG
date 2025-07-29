# vector.py

import os
import shutil
from dotenv import load_dotenv

# ─── Load environment variables ──────────────────────────────────────────────
load_dotenv()
DB_LOCATION     = os.getenv("DB_LOCATION", "./chrome_langchain_db")

# ─── Collection and data folder locations ─────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "restaurant_reviews")
DATA_FOLDER     = os.getenv("DATA_FOLDER", "./data")

# ─── Embedding service configuration ───────────────────────────────────────
EMBEDDING_HOST   = os.getenv("EMBEDDING_HOST", "http://localhost:12434")
EMBEDDING_ENGINE = os.getenv("EMBEDDING_ENGINE", "llama.cpp")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "ai/mxbai-embed-large:latest")

# ─── Toggle to rebuild the vector store on each run ────────────────────────
REBUILD_FLAG     = os.getenv("REBUILD_VECTOR", "false").lower() in ("1", "true", "yes")

# ─── Prepare or rebuild local ChromaDB files ───────────────────────────────
if REBUILD_FLAG and os.path.exists(DB_LOCATION):
    shutil.rmtree(DB_LOCATION, ignore_errors=True)
add_documents = not os.path.exists(DB_LOCATION)

# ─── Standard imports ──────────────────────────────────────────────────────
import pandas as pd
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    PythonLoader,
    TextLoader
)
# ChromaDB imports no longer needed with new client format

# ─── Custom Embeddings via HTTP ────────────────────────────────────────────
class LocalEmbeddings:
    def __init__(self, host: str, engine: str, model: str):
        self.url = f"{host}/engines/{engine}/v1/embeddings"
        self.model = model

    def _embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "input": text}
        resp = requests.post(
            self.url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [{}])[0].get("embedding", [])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

# Instantiate the embedding client
embeddings = LocalEmbeddings(
    host=EMBEDDING_HOST,
    engine=EMBEDDING_ENGINE,
    model=EMBEDDING_MODEL
)

# ─── Helper functions for loading various file types ────────────────────────
def load_csv(path: str) -> list[Document]:
    df = pd.read_csv(path)
    docs: list[Document] = []
    for i, row in df.iterrows():
        content = f"{row.get('Title','')} {row.get('Review','')}"
        metadata = {"source": path, "rating": row.get('Rating'), "date": row.get('Date')}
        docs.append(Document(page_content=content, metadata=metadata, id=f"csv-{i}"))
    return docs


def load_python(path: str) -> list[Document]:
    try:
        return PythonLoader(path).load()
    except Exception:
        return TextLoader(path).load()


def load_yaml(path: str) -> list[Document]:
    return TextLoader(path).load()


def load_markdown(path: str) -> list[Document]:
    try:
        return UnstructuredMarkdownLoader(path).load()
    except Exception:
        return TextLoader(path).load()


def load_html(path: str) -> list[Document]:
    try:
        return UnstructuredHTMLLoader(path).load()
    except Exception:
        return TextLoader(path).load()

# ─── Aggregate all documents from DATA_FOLDER ───────────────────────────────
documents: list[Document] = []
if add_documents:
    for subdir, loader in (
        ("csv", load_csv),
        ("py", load_python),
        ("yaml", load_yaml),
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
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")

# ─── Initialize Chroma client and build/open vector store ──────────────────
# Use the new ChromaDB client format (persistent client)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DB_LOCATION
)

if add_documents and documents:
    vector_store.add_documents(documents=documents)

# ─── Retriever ─────────────────────────────────────────────────────────────
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
