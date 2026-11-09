# RRR Support Agent: Milestones A–B (chat, voice, RAG + human handoff)

A customer writes a machine problem in English, Telugu or Hindi. The system finds the
matching guide, answers from it only, and hands off to a human when it is not sure.
All guides in `data/sample_docs/` are FAKE sample data. Have a native speaker and an
RRR technician review the Telugu/Hindi text before showing real customers.

## Setup (once)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
cd frontend && npm install && cd ..
```

### Optional: local bge-m3 only
BGE-M3 needs PyTorch and FlagEmbedding. It is intentionally optional: mock/hash and Gemini
configurations do not import it. Only when you select `EMBED_PROVIDER=bge-m3`, install its local
requirements:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install FlagEmbedding
```
This project environment already uses mock/hash for tests; do not install or upgrade packages for
that workflow.

## Milestone A provider and storage choices
Set providers only in `.env`, never in code: `LLM_PROVIDER=mock|gemini|openai_compat` and
`EMBED_PROVIDER=hash|gemini|bge-m3`. Mock is deterministic and makes no network calls. NumpyStore
is the cloud-default small persistent vector index; `VECTOR_STORE=chroma` remains the optional
local development route. `DATABASE_URL` takes priority for cloud Postgres, while `SQLITE_URL`
continues to work locally.

Embedding similarity values are not comparable between hash, Gemini, and BGE-M3. Set the matching
`CONFIDENCE_THRESHOLD_HASH`, `CONFIDENCE_THRESHOLD_GEMINI`, or
`CONFIDENCE_THRESHOLD_BGE_M3`, then re-tune after every embedding-provider switch:
```
python -m tests.run_eval
```
If the persisted Numpy index reports a different embedding model, rebuild it before searching.

## Run
```
python -m backend.ingest             # builds the search index (first run downloads bge-m3, ~2 GB)
uvicorn backend.main:app --reload    # backend on http://127.0.0.1:8000
cd frontend && npm run dev           # chat UI on http://localhost:5173
```
With no `LLM_API_KEY` in `.env` the app runs in offline mode: it returns the best guide text
instead of an LLM-written answer. Add a key and `LLM_MODEL` for real answers.

## Test
```
python -m pytest                     # fast offline tests (mock/hash, no download)
python -m tests.run_eval             # 20 questions with the REAL model; tune CONFIDENCE_THRESHOLD
```

## How a question flows (read the files in this order)
1. `backend/config.py`: settings from `.env`.
2. `backend/language.py`: Telugu/Hindi/English detection by Unicode script, and a tokenizer.
3. `backend/embeddings.py`: text to vectors (bge-m3, or the tiny `hash` test embedder).
4. `backend/ingest.py`: guide files to chunks to vectors to Chroma.
5. `backend/rag.py`: hybrid search (vectors + BM25, merged by Reciprocal Rank Fusion), confidence
   check, LLM answer restricted to the retrieved text, handoff rule.
6. `backend/db.py`: chat history, unanswered-question log, handoff tickets (SQLite).
7. `backend/main.py`: `/chat`, `/voice`, and `/health` endpoints.
8. `backend/providers/`: selected LLM, embedding, STT and TTS adapters with retries and fallbacks.
9. `frontend/src/App.jsx`: text chat plus browser microphone recording.

## Voice input (Milestone B)
The **Voice** button records a maximum of 30 seconds, converts the browser recording to mono, 16 kHz, 16-bit WAV in the browser, and submits it to `POST /voice`. The endpoint accepts only WAV input (maximum 2 MB / 30 seconds), transcribes it, uses the same grounded RAG and handoff flow as `/chat`, and returns response text plus base64 audio for playback. Text chat remains available if microphone access or voice processing fails.

The default `STT_PROVIDER=mock` and `TTS_PROVIDER=mock` work fully offline. Select `gemini` or `gnani` STT, or `edge`/`gnani` TTS, only through private `.env` settings. The configured `*_FALLBACK` is tried after timeout/retry failure. Gemini TTS is intentionally deferred because its SDK audio output has not been verified here.

Generated response audio is locally cached in `AUDIO_CACHE_PATH`; entries are exact matches of provider, language, voice/cache key, and answer text. Do not commit this cache. To manually validate a private Gnani credential and WAV, run:
```cmd
python scripts/smoke_gnani.py path\to\recording.wav
```

## Handoff rule
A question goes to a human when the best match scores below `CONFIDENCE_THRESHOLD`, or the LLM
replies `HANDOFF`. Each handoff creates a ticket (with the last messages) and is logged in
`unanswered_questions`, which tells you which guide or clip to add next.

## Milestones 2–5
- **Media diagnosis:** `POST /diagnose` accepts a customer message and one image or video (max 5 MB). The system keeps answers grounded exclusively in indexed guides; supplied media is stored as technician evidence rather than being represented as an unverified machine diagnosis.
- **Live support workflow:** media evidence and the complete chat summary are attached to an automatically opened technician ticket. This provides an asynchronous, auditable alternative to an unsafe unauthenticated live-video service.
- **Technician dashboard API:** protected endpoints list, read, update, and retrieve evidence for tickets. Set `TECHNICIAN_TOKEN` and send `Authorization: Bearer <token>`; endpoints are disabled while the token is blank.
- **Alerts:** set `NOTIFICATION_WEBHOOK_URL` to receive a `ticket.created` HTTPS webhook. This generic integration supports the organisation's approved Telegram, email, or ticketing automation without storing third-party credentials in code.
- **Reranking:** hybrid retrieval now includes configurable lexical reranking (`RERANK_ENABLED`) for exact symptoms and error codes.

The single React customer UI remains intentionally separate from the protected technician API. A production technician web dashboard should be deployed behind the organisation's SSO/identity provider rather than embedding an administrator login in the public customer app.
