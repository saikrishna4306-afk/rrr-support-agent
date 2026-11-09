"""Small persistent NumPy store for cloud-scale guide collections (brute-force cosine)."""
import asyncio
import json
from pathlib import Path

import numpy as np

from .base import StoredDocument, VectorStore


class NumpyStore(VectorStore):
    """Writes one NPZ plus JSON metadata; refuses vectors from another embedding model."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.vectors_file = self.path / "vectors.npz"
        self.docs_file = self.path / "documents.json"

    def _load(self, embedding_model: str) -> list[StoredDocument]:
        if not self.docs_file.exists() or not self.vectors_file.exists():
            return []
        payload = json.loads(self.docs_file.read_text(encoding="utf-8"))
        stored_model = payload["embedding_model"]
        if stored_model != embedding_model:
            raise ValueError("Index was built with embedding model " + repr(stored_model) +
                             "; current model is " + repr(embedding_model) + ". Re-ingest first.")
        vectors = np.load(self.vectors_file)["vectors"]
        return [StoredDocument(item["id"], item["text"], item["metadata"], vector.tolist())
                for item, vector in zip(payload["documents"], vectors)]

    def _replace(self, documents: list[StoredDocument], embedding_model: str) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        vectors = np.asarray([doc.embedding for doc in documents], dtype=np.float32)
        np.savez_compressed(self.vectors_file, vectors=vectors)
        payload = {"embedding_model": embedding_model, "documents": [
            {"id": doc.id, "text": doc.text, "metadata": doc.metadata} for doc in documents
        ]}
        self.docs_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    async def replace_all(self, documents: list[StoredDocument], embedding_model: str) -> None:
        await asyncio.to_thread(self._replace, documents, embedding_model)

    async def all_documents(self, embedding_model: str) -> list[StoredDocument]:
        return await asyncio.to_thread(self._load, embedding_model)

    async def search(self, embedding: list[float], k: int, machine_model: str | None,
                     embedding_model: str) -> list[tuple[StoredDocument, float]]:
        documents = await self.all_documents(embedding_model)
        allowed = {machine_model, "ALL"} if machine_model else None
        candidates = [doc for doc in documents if not allowed or doc.metadata.get("machine_model") in allowed]
        if not candidates:
            return []
        matrix = np.asarray([doc.embedding for doc in candidates], dtype=np.float32)
        query = np.asarray(embedding, dtype=np.float32)
        scores = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-12)
        order = np.argsort(-scores)[:k]
        return [(candidates[int(i)], float(scores[int(i)])) for i in order]
