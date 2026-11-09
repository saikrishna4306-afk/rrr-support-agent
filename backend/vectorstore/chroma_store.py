"""Optional Chroma adapter; retained for local development compatibility."""
import asyncio

from .base import StoredDocument, VectorStore
from ..ingest import COLLECTION


class ChromaStore(VectorStore):
    """Adapts a Chroma collection to the common async VectorStore contract."""

    def __init__(self, collection) -> None:
        self.collection = collection

    async def replace_all(self, documents: list[StoredDocument], embedding_model: str) -> None:
        def write() -> None:
            existing = self.collection.get(include=[])["ids"]
            if existing:
                self.collection.delete(ids=existing)
            self.collection.add(ids=[d.id for d in documents], documents=[d.text for d in documents],
                metadatas=[{**d.metadata, "embedding_model": embedding_model} for d in documents],
                embeddings=[d.embedding for d in documents])
        await asyncio.to_thread(write)

    async def all_documents(self, embedding_model: str) -> list[StoredDocument]:
        def read() -> list[StoredDocument]:
            data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            documents = []
            for identifier, text, metadata, vector in zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"]):
                if metadata.get("embedding_model", embedding_model) != embedding_model:
                    raise ValueError("Index embedding model differs; re-ingest first.")
                metadata.pop("embedding_model", None)
                documents.append(StoredDocument(identifier, text, metadata, vector))
            return documents
        return await asyncio.to_thread(read)

    async def search(self, embedding: list[float], k: int, machine_model: str | None,
                     embedding_model: str) -> list[tuple[StoredDocument, float]]:
        # Existing RAG continues to use Chroma directly during this compatibility milestone.
        documents = await self.all_documents(embedding_model)
        return [(doc, 0.0) for doc in documents[:k]]
