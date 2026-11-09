import asyncio

import pytest

from backend.vectorstore.base import StoredDocument
from backend.vectorstore.numpy_store import NumpyStore


def test_numpy_store_persists_searches_and_checks_embedding_model(tmp_path):
    """Use asyncio.run because this pytest installation has no async-test plugin."""
    async def scenario():
        store = NumpyStore(str(tmp_path / "index"))
        docs = [
            StoredDocument("one", "needle", {"machine_model": "ALL"}, [1.0, 0.0]),
            StoredDocument("two", "bobbin", {"machine_model": "X1"}, [0.0, 1.0]),
        ]
        await store.replace_all(docs, "hash-512-v1")
        found = await store.search([1.0, 0.0], 1, None, "hash-512-v1")
        assert found[0][0].id == "one"
        filtered = await store.search([0.0, 1.0], 1, "X1", "hash-512-v1")
        assert filtered[0][0].id == "two"
        with pytest.raises(ValueError, match="Re-ingest"):
            await store.all_documents("another-model")
    asyncio.run(scenario())
