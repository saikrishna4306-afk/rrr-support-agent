"""Reads the guides in data/sample_docs, splits them into chunks, embeds them, stores them in Chroma.

Run it once (and again whenever documents change):
    python -m backend.ingest

Document format (a few "key: value" lines, then a line with ---, then the text):
    title: Upper thread keeps breaking
    machine_model: ALL
    language: en
    ---
    ...guide text...
"""
import re
from pathlib import Path

import chromadb

from .config import settings
from .embeddings import embed
from .providers.registry import call_provider, get_embedding_provider
from .vectorstore.base import StoredDocument
from .vectorstore.numpy_store import NumpyStore

COLLECTION = "rrr_docs"


def get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_path)
    # cosine = compare vectors by angle; distance 0 means identical meaning.
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def parse_doc(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    head, _, body = raw.partition("\n---\n")
    meta = {}
    for line in head.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return {"meta": meta, "body": body.strip()}


def chunk_text(body: str, max_chars: int = 700) -> list[str]:
    """Group paragraphs together until a chunk would exceed max_chars."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def ingest(reset: bool = True) -> int:
    if reset:  # start clean so deleted/renamed files do not linger in the index
        client = chromadb.PersistentClient(path=settings.chroma_path)
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = get_collection()

    ids, texts, metas = [], [], []
    for path in sorted(Path(settings.docs_path).glob("*.md")):
        doc = parse_doc(path)
        title = doc["meta"].get("title", path.stem)
        for i, chunk in enumerate(chunk_text(doc["body"])):
            ids.append(f"{path.name}-{i}")
            texts.append(f"{title}\n{chunk}")  # the title travels with every chunk
            metas.append({
                "title": title,
                "source": path.name,
                "machine_model": doc["meta"].get("machine_model", "ALL"),
                "language": doc["meta"].get("language", "en"),
            })
    if not ids:
        return 0
    collection.add(ids=ids, documents=texts, embeddings=embed(texts), metadatas=metas)
    return len(ids)


async def ingest_numpy(reset: bool = True) -> int:
    """Async cloud-default ingestion through the provider and VectorStore interfaces."""
    provider = get_embedding_provider()
    documents: list[StoredDocument] = []
    for path in sorted(Path(settings.docs_path).glob("*.md")):
        doc = parse_doc(path)
        title = doc["meta"].get("title", path.stem)
        for number, chunk in enumerate(chunk_text(doc["body"])):
            documents.append(StoredDocument(
                id=f"{path.name}-{number}", text=f"{title}\n{chunk}",
                metadata={"title": title, "source": path.name,
                          "machine_model": doc["meta"].get("machine_model", "ALL"),
                          "language": doc["meta"].get("language", "en")}, embedding=[]))
    vectors = await call_provider(lambda: provider.embed([doc.text for doc in documents]))
    for document, vector in zip(documents, vectors):
        document.embedding = vector
    await NumpyStore(settings.numpy_path).replace_all(documents, provider.model_name)
    return len(documents)


if __name__ == "__main__":
    # The command remains synchronous for Windows CLI use; API code never calls asyncio.run().
    if settings.vector_store == "numpy":
        import asyncio
        print(f"Indexed {asyncio.run(ingest_numpy())} chunks into {settings.numpy_path}")
    else:
        print(f"Indexed {ingest()} chunks into {settings.chroma_path}")
