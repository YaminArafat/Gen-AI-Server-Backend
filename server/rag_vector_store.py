from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_openai import OpenAIEmbeddings

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", ROOT_DIR / "server" / "chroma_db"))


def _get_embeddings() -> OllamaEmbeddings:
    model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama2")
    return OllamaEmbeddings(
        model=model_name,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


# def _get_embeddings() -> OpenAIEmbeddings:
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise EnvironmentError(
#             "OPENAI_API_KEY is not configured.\n"
#             "Set OPENAI_API_KEY in your environment or .env file before using RAG."
#         )
#     return OpenAIEmbeddings()


def _load_dataset_documents(dataset_paths: List[Path]) -> List[Document]:
    documents: List[Document] = []
    for path in dataset_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source = path.name
                documents.append(
                    Document(
                        page_content=json.dumps(record, ensure_ascii=False),
                        metadata={"source": source},
                    )
                )
    return documents


def build_vector_store(dataset_paths: List[Path] | None = None, persist_dir: Path | None = None) -> Chroma:
    persist_dir = Path(persist_dir or PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = _get_embeddings()
    dataset_paths = dataset_paths or sorted(ROOT_DIR.glob("datasets/*.jsonl"))
    documents = _load_dataset_documents(dataset_paths)
    if not documents:
        raise RuntimeError("No documents were available to ingest. Add JSONL files to the datasets/folder.")
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="config_docs",
    )
    store.persist()
    return store


def load_vector_store(persist_dir: Path | None = None) -> Chroma:
    persist_dir = Path(persist_dir or PERSIST_DIR)
    if not persist_dir.exists():
        raise RuntimeError("Chroma vector store does not exist. Call build_vector_store() first.")
    embeddings = _get_embeddings()
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="config_docs",
    )


def query_rag(query: str, k: int = 4, persist_dir: Path | None = None) -> list[dict]:
    store = load_vector_store(persist_dir=persist_dir)
    docs = store.similarity_search(query, k=k)
    return [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata or {}),
        }
        for doc in docs
    ]
