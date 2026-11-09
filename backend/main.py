"""The web API. Start it with:  uvicorn backend.main:app --reload"""
import asyncio
import base64
import io
import mimetypes
import secrets
import uuid
import wave
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import db
from .config import settings
from .language import detect_language
from .notifications import notify_new_ticket
from .observability import log_event, new_request_id, timed
from .rag import answer_question, get_retriever, reset_retriever
from .ingest import ingest_numpy
from .vectorstore.numpy_store import NumpyStore
from .audio_cache import load as load_cached_audio, save as save_cached_audio
from .providers.registry import call_provider, get_embedding_provider, get_stt_provider, get_tts_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(db.init_db)  # SQLAlchemy is synchronous.
    if settings.vector_store == "numpy":
        provider = get_embedding_provider()
        store = NumpyStore(settings.numpy_path)
        if not await store.all_documents(provider.model_name):
            await ingest_numpy()
        reset_retriever()
    yield


app = FastAPI(title="RRR Support Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_wav(audio: bytes) -> None:
    """Reject malformed or overlong WAV input before an external STT provider sees it."""
    if len(audio) > settings.max_audio_bytes:
        raise HTTPException(413, f"Audio must be at most {settings.max_audio_bytes} bytes")
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError) as error:
        raise HTTPException(422, "Audio must be a valid WAV file") from error
    if duration > settings.max_audio_seconds:
        raise HTTPException(422, f"Audio must be at most {settings.max_audio_seconds:g} seconds")


def require_technician(authorization: str | None = Header(None)) -> None:
    """Protect dashboard APIs. An empty token intentionally disables their access."""
    expected = settings.technician_token
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not expected or not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "A valid technician bearer token is required")


async def _save_evidence(upload: UploadFile | None) -> tuple[str, str]:
    """Store one bounded image/video evidence file with an opaque server-side name."""
    if upload is None:
        return "", ""
    if not upload.content_type or not (upload.content_type.startswith("image/") or upload.content_type.startswith("video/")):
        raise HTTPException(415, "Evidence must be an image or video")
    data = await upload.read(settings.max_evidence_bytes + 1)
    if len(data) > settings.max_evidence_bytes:
        raise HTTPException(413, f"Evidence must be at most {settings.max_evidence_bytes} bytes")
    suffix = Path(upload.filename or "").suffix[:10] or mimetypes.guess_extension(upload.content_type) or ""
    path = Path(settings.evidence_path) / f"{uuid.uuid4().hex}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path), upload.content_type


async def _with_fallback(kind: str, call):
    """Use configured fallback only after the selected STT/TTS provider fails."""
    try:
        return await call()
    except Exception:
        fallback = settings.stt_fallback if kind == "stt" else settings.tts_fallback
        selected = settings.stt_provider if kind == "stt" else settings.tts_provider
        if not fallback or fallback == selected:
            raise
        return await call(fallback)


class TicketUpdate(BaseModel):
    status: str = Field(pattern="^(open|in_progress|resolved|closed)$")
    technician_note: str = Field(default="", max_length=2000)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=64)
    message: str = Field(min_length=1, max_length=1000)
    machine_model: str | None = None


@app.get("/health")
def health():
    mode = "mock" if settings.llm_provider == "mock" else settings.llm_provider
    return {"status": "ok", "chunks_indexed": len(get_retriever().ids), "mode": mode,
            "embedding_provider": settings.active_embed_provider, "vector_store": settings.vector_store}


@app.post("/chat")
async def chat(req: ChatRequest):
    """Preserve the old API while safely isolating Milestone 1 blocking work."""
    request_id, timings = new_request_id(), {}

    def blocking_chat():
        language = detect_language(req.message)
        db.save_message(req.session_id, "user", req.message, language)
        result = answer_question(req.message, req.machine_model)
        ticket_id = None
        if result.handoff:
            db.log_unanswered(req.message, language, result.confidence)
            ticket_id = db.create_ticket(req.session_id, req.message)
        db.save_message(req.session_id, "assistant", result.answer, language)
        return result, ticket_id

    with timed(timings, "chat"):
        result, ticket_id = await asyncio.to_thread(blocking_chat)
    log_event("chat_completed", request_id, handoff=result.handoff)
    return {
        "answer": result.answer,
        "language": result.language,
        "confidence": round(result.confidence, 3),
        "handoff": result.handoff,
        "ticket_id": ticket_id,
        "sources": result.sources,
        "timings": timings,
    }


@app.post("/diagnose")
async def diagnose(message: str = Form(..., min_length=1, max_length=1000), session_id: str = Form(...),
                   machine_model: str | None = Form(None), evidence: UploadFile | None = File(None)):
    """Accept photo/video evidence for a human handoff; retrieval remains grounded in written guides."""
    if not 8 <= len(session_id) <= 64:
        raise HTTPException(422, "session_id must contain 8 to 64 characters")
    evidence_path, evidence_type = await _save_evidence(evidence)

    def blocking_diagnosis():
        language = detect_language(message)
        db.save_message(session_id, "user", message, language)
        result = answer_question(message, machine_model)
        ticket_id = None
        if result.handoff or evidence_path:
            db.log_unanswered(message, language, result.confidence)
            ticket_id = db.create_ticket(session_id, message, evidence_path, evidence_type)
        db.save_message(session_id, "assistant", result.answer, result.language)
        return result, ticket_id

    result, ticket_id = await asyncio.to_thread(blocking_diagnosis)
    if ticket_id:
        try:
            await notify_new_ticket(ticket_id, message)
        except Exception:
            log_event("ticket_notification_failed", new_request_id()[0], ticket_id=ticket_id)
    return {"answer": result.answer, "language": result.language, "confidence": round(result.confidence, 3),
            "handoff": result.handoff or bool(evidence_path), "ticket_id": ticket_id, "sources": result.sources}


@app.get("/technician/tickets", dependencies=[Depends(require_technician)])
async def technician_tickets(status: str | None = None):
    if status and status not in {"open", "in_progress", "resolved", "closed"}:
        raise HTTPException(422, "Invalid ticket status")
    return {"tickets": await asyncio.to_thread(db.list_tickets, status)}


@app.get("/technician/tickets/{ticket_id}", dependencies=[Depends(require_technician)])
async def technician_ticket(ticket_id: int):
    ticket = await asyncio.to_thread(db.get_ticket, ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    return ticket


@app.patch("/technician/tickets/{ticket_id}", dependencies=[Depends(require_technician)])
async def technician_update_ticket(ticket_id: int, update: TicketUpdate):
    ticket = await asyncio.to_thread(db.update_ticket, ticket_id, update.status, update.technician_note)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    return ticket


@app.get("/technician/tickets/{ticket_id}/evidence", dependencies=[Depends(require_technician)])
async def technician_ticket_evidence(ticket_id: int):
    ticket = await asyncio.to_thread(db.get_ticket, ticket_id)
    if not ticket or not ticket["has_evidence"]:
        raise HTTPException(404, "Evidence not found")
    with db.SessionLocal() as session:
        row = session.get(db.Ticket, ticket_id)
        path = Path(row.evidence_path)
        if not path.is_file():
            raise HTTPException(404, "Evidence file not found")
        return FileResponse(path, media_type=row.evidence_type, filename=path.name)



@app.post("/voice")
async def voice(audio: UploadFile = File(...), session_id: str = Form(...),
                machine_model: str | None = Form(None)):
    """Transcribe a browser WAV, reuse grounded support chat, then speak its response."""
    if not 8 <= len(session_id) <= 64:
        raise HTTPException(422, "session_id must contain 8 to 64 characters")
    if audio.content_type not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
        raise HTTPException(415, "Only WAV audio is accepted")
    raw_audio = await audio.read(settings.max_audio_bytes + 1)
    _validate_wav(raw_audio)
    request_id, timings = new_request_id(), {}

    async def transcribe(selected: str | None = None):
        provider = get_stt_provider(selected)
        with timed(timings, "stt"):
            return await call_provider(lambda: provider.transcribe(raw_audio))

    try:
        transcript = await _with_fallback("stt", transcribe)
    except Exception as error:
        log_event("voice_stt_failed", request_id, provider=settings.stt_provider)
        raise HTTPException(503, "Speech recognition is temporarily unavailable") from error
    if not transcript.text or len(transcript.text) > 1000:
        raise HTTPException(422, "Speech recognition did not return a usable transcript")

    def support_answer():
        language = transcript.language or detect_language(transcript.text)
        db.save_message(session_id, "user", transcript.text, language)
        result = answer_question(transcript.text, machine_model)
        ticket_id = None
        if result.handoff:
            db.log_unanswered(transcript.text, language, result.confidence)
            ticket_id = db.create_ticket(session_id, transcript.text)
        db.save_message(session_id, "assistant", result.answer, result.language)
        return result, ticket_id

    with timed(timings, "retrieve"):
        result, ticket_id = await asyncio.to_thread(support_answer)

    async def synthesize(selected: str | None = None):
        selected = selected or settings.tts_provider
        cached = load_cached_audio(result.answer, "default", result.language, selected)
        if cached:
            timings["tts"] = 0.0
            return cached
        provider = get_tts_provider(selected)
        with timed(timings, "tts"):
            spoken = await call_provider(lambda: provider.synthesize(result.answer, result.language))
        # Cache key represents the configured provider, language, and exact text. The
        # adapter result retains its actual voice name in the response metadata.
        save_cached_audio(result.answer, result.language, selected,
                         type(spoken)(spoken.data, spoken.mime_type, "default"))
        return spoken

    try:
        spoken = await _with_fallback("tts", synthesize)
    except Exception as error:
        log_event("voice_tts_failed", request_id, provider=settings.tts_provider)
        raise HTTPException(503, "Speech synthesis is temporarily unavailable") from error
    log_event("voice_completed", request_id, handoff=result.handoff)
    return {"transcript": transcript.text, "language": result.language, "answer_text": result.answer,
            "audio": base64.b64encode(spoken.data).decode("ascii"), "audio_mime_type": spoken.mime_type,
            "video": None, "fault_id": None, "sources": result.sources,
            "confidence": round(result.confidence, 3), "handoff": result.handoff,
            "ticket_id": ticket_id, "timings": timings}
