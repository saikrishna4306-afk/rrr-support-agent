"""Turns text into vectors (embeddings) so we can search by meaning.

Two backends:
  bge-m3 : real multilingual model (Telugu/Hindi/English). Slow on CPU, ~2 GB download.
  hash   : tiny bag-of-words embedder with no download. Only for fast offline tests.
"""
import hashlib
import math

from .config import settings
from .language import tokenize

_model = None  # loaded once, on first use (loading takes a while)


def _hash_embed(texts: list[str], dim: int = 512) -> list[list[float]]:
    out = []
    for text in texts:
        vec = [0.0] * dim
        for token in tokenize(text):
            slot = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
            vec[slot] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])  # unit length -> dot product = cosine
    return out


def _bge_embed(texts: list[str]) -> list[list[float]]:
    """Run BGE-M3 synchronously; provider code moves this blocking work to a thread."""
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel  # optional heavy dependency; lazy by design
        _model = BGEM3FlagModel(settings.embed_model, use_fp16=False)
    vectors = _model.encode(texts, batch_size=8, max_length=512)["dense_vecs"]
    return [[float(x) for x in v] for v in vectors]


def embed(texts: list[str]) -> list[list[float]]:
    """Milestone 1 synchronous compatibility API used by legacy scripts/tests."""
    if settings.active_embed_provider == "hash":
        return _hash_embed(texts)
    return _bge_embed(texts)
