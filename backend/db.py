"""SQLite database (SQLAlchemy): chat history, unanswered questions, handoff tickets."""
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))          # "user" or "assistant"
    text: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(4), default="en")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UnansweredQuestion(Base):
    """Every question the AI could not answer. This list tells you which clip/guide to add next."""
    __tablename__ = "unanswered_questions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(4))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Ticket(Base):
    """A request waiting for a human technician."""
    __tablename__ = "tickets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)              # last few messages, for the technician
    status: Mapped[str] = mapped_column(String(16), default="open")
    technician_note: Mapped[str] = mapped_column(Text, default="")
    evidence_path: Mapped[str] = mapped_column(String(512), default="")
    evidence_type: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


_database_url = settings.active_database_url
_connect_args = {"check_same_thread": False} if _database_url.startswith("sqlite") else {}
engine = create_engine(_database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def init_db() -> None:
    """Create new tables and add Milestones 2-5 ticket columns to existing local DBs."""
    Base.metadata.create_all(engine)
    existing = {column["name"] for column in inspect(engine).get_columns("tickets")}
    additions = {"technician_note": "TEXT NOT NULL DEFAULT ''", "evidence_path": "VARCHAR(512) NOT NULL DEFAULT ''",
                 "evidence_type": "VARCHAR(64) NOT NULL DEFAULT ''", "updated_at": "DATETIME"}
    with engine.begin() as connection:
        for name, sql_type in additions.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE tickets ADD COLUMN {name} {sql_type}"))
        if "updated_at" not in existing:
            connection.execute(text("UPDATE tickets SET updated_at = created_at WHERE updated_at IS NULL"))


def save_message(session_id: str, role: str, text: str, language: str) -> None:
    with SessionLocal() as s:
        s.add(Message(session_id=session_id, role=role, text=text, language=language))
        s.commit()


def log_unanswered(question: str, language: str, confidence: float) -> None:
    with SessionLocal() as s:
        s.add(UnansweredQuestion(question=question, language=language, confidence=confidence))
        s.commit()


def create_ticket(session_id: str, question: str, evidence_path: str = "", evidence_type: str = "") -> int:
    with SessionLocal() as s:
        recent = s.scalars(
            select(Message).where(Message.session_id == session_id)
            .order_by(Message.id.desc()).limit(6)
        ).all()
        summary = "\n".join(f"{m.role}: {m.text}" for m in reversed(recent))
        ticket = Ticket(session_id=session_id, question=question, summary=summary,
                        evidence_path=evidence_path, evidence_type=evidence_type)
        s.add(ticket)
        s.commit()
        return ticket.id


def list_tickets(status: str | None = None) -> list[dict]:
    with SessionLocal() as s:
        statement = select(Ticket).order_by(Ticket.updated_at.desc(), Ticket.id.desc())
        if status:
            statement = statement.where(Ticket.status == status)
        return [_ticket_dict(ticket) for ticket in s.scalars(statement)]


def get_ticket(ticket_id: int) -> dict | None:
    with SessionLocal() as s:
        ticket = s.get(Ticket, ticket_id)
        return _ticket_dict(ticket) if ticket else None


def update_ticket(ticket_id: int, status: str, technician_note: str) -> dict | None:
    with SessionLocal() as s:
        ticket = s.get(Ticket, ticket_id)
        if not ticket:
            return None
        ticket.status = status
        ticket.technician_note = technician_note
        ticket.updated_at = _now()
        s.commit()
        return _ticket_dict(ticket)


def _ticket_dict(ticket: Ticket) -> dict:
    return {"id": ticket.id, "session_id": ticket.session_id, "question": ticket.question,
            "summary": ticket.summary, "status": ticket.status, "technician_note": ticket.technician_note,
            "has_evidence": bool(ticket.evidence_path), "evidence_type": ticket.evidence_type,
            "created_at": ticket.created_at.isoformat(), "updated_at": ticket.updated_at.isoformat()}
