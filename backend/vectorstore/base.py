"""Async vector-store contract."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class StoredDocument:
    id: str
    text: str
    metadata: dict[str, str]
    embedding: list[float]


class VectorStore(ABC):
    """Stores vectors and searches them by cosine similarity."""

    @abstractmethod
    async def replace_all(self, documents: list[StoredDocument], embedding_model: str) -> None: ...

    @abstractmethod
    async def all_documents(self, embedding_model: str) -> list[StoredDocument]: ...

    @abstractmethod
    async def search(self, embedding: list[float], k: int, machine_model: str | None,
                     embedding_model: str) -> list[tuple[StoredDocument, float]]: ...
