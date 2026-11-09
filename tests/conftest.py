"""Runs before any test. Points the app at throw-away storage and the tiny 'hash' embedder,
so the unit tests are fast, free and need no model download."""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["EMBED_BACKEND"] = "hash"
os.environ["EMBED_PROVIDER"] = "hash"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["VECTOR_STORE"] = "chroma"  # Preserve existing Milestone 1 retrieval assertions.
os.environ["CHROMA_PATH"] = os.path.join(_tmp, "chroma")
os.environ["NUMPY_PATH"] = os.path.join(_tmp, "numpy")
os.environ["SQLITE_URL"] = "sqlite:///" + os.path.join(_tmp, "test.db")
os.environ["LLM_API_KEY"] = ""             # force offline mode
os.environ["CONFIDENCE_THRESHOLD"] = "0.2"  # the hash embedder scores lower than bge-m3
