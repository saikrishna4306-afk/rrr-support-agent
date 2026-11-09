"""Central settings. Values come from the .env file (never hard-code secrets).

pydantic-settings reads each field from an environment variable with the same
name (LLM_API_KEY, EMBED_BACKEND, ...). Real environment variables win over .env.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # --- Provider selection. "mock" is deterministic and never uses the network. ---
    llm_provider: str = "mock"          # mock | gemini | openai_compat
    llm_fallback: str = ""
    stt_provider: str = "mock"           # mock | gemini | gnani
    stt_fallback: str = "gemini"
    tts_provider: str = "mock"           # mock | edge | gemini | gnani
    tts_fallback: str = "edge"
    embed_provider: str = ""             # gemini | bge-m3 | hash; empty keeps EMBED_BACKEND
    vector_store: str = "numpy"          # numpy (cloud default) | chroma (optional local dev)
    provider_timeout_seconds: float = 30.0
    provider_max_retries: int = 2

    # --- LLM: legacy LLM_* names remain supported for OpenAI-compatible providers. ---
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = ""
    gemini_api_key: str = ""
    gemini_llm_model: str = ""
    gemini_embed_model: str = ""
    gemini_stt_model: str = ""
    gemini_tts_model: str = ""

    # --- Gnani REST providers (leave the key blank until you receive one). ---
    gnani_api_key: str = ""
    gnani_base_url: str = "https://api.vachana.ai"
    gnani_stt_model: str = "prisma-v3.0"
    gnani_tts_model: str = "timbre-v2.5"
    gnani_tts_voice_te: str = ""
    gnani_tts_voice_hi: str = ""
    gnani_tts_voice_en: str = ""
    edge_tts_voice_te: str = "te-IN-ShrutiNeural"
    edge_tts_voice_hi: str = "hi-IN-SwaraNeural"
    edge_tts_voice_en: str = "en-IN-NeerjaNeural"
    gnani_stt_unit_cost: float = 0.0     # Fill in only after confirming your commercial rate.
    gnani_tts_unit_cost: float = 0.0

    # --- Retrieval ---
    embed_backend: str = "hash"          # legacy name; used when EMBED_PROVIDER is empty
    embed_model: str = "BAAI/bge-m3"
    confidence_threshold: float = 0.45    # legacy fallback threshold
    confidence_threshold_hash: float = 0.20
    confidence_threshold_gemini: float = 0.45
    confidence_threshold_bge_m3: float = 0.45
    top_k: int = 3

    # --- Storage ---
    chroma_path: str = str(ROOT / "chroma_db")
    numpy_path: str = str(ROOT / "numpy_index")
    sqlite_url: str = f"sqlite:///{ROOT / 'support.db'}"  # legacy local database name
    database_url: str = ""               # cloud Postgres or SQLite; takes precedence when set
    docs_path: str = str(ROOT / "data" / "sample_docs")
    allowed_origins: str = "http://localhost:5173"
    audio_cache_path: str = str(ROOT / "audio_cache")
    max_audio_bytes: int = 2 * 1024 * 1024
    max_audio_seconds: float = 30.0
    evidence_path: str = str(ROOT / "evidence")
    max_evidence_bytes: int = 5 * 1024 * 1024
    technician_token: str = ""           # Required for technician-only ticket APIs.
    notification_webhook_url: str = ""   # Optional HTTPS webhook called for new handoffs.
    notification_webhook_token: str = ""
    rerank_enabled: bool = True

    @property
    def active_embed_provider(self) -> str:
        """Prefer the new setting, while accepting Milestone 1 EMBED_BACKEND."""
        return (self.embed_provider or self.embed_backend).lower()

    @property
    def active_confidence_threshold(self) -> float:
        """Embedding spaces differ, so each provider has an independently tuned cutoff."""
        return {
            "hash": self.confidence_threshold_hash,
            "gemini": self.confidence_threshold_gemini,
            "bge-m3": self.confidence_threshold_bge_m3,
        }.get(self.active_embed_provider, self.confidence_threshold)

    @property
    def active_database_url(self) -> str:
        return self.database_url or self.sqlite_url

    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")


settings = Settings()
