"""FastAPI service exposing MITRA over HTTP, plus the static chat UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from . import safety
from .config import settings
from .orchestrator import Orchestrator
from .schemas import AgentResponse, ChatRequest, MoodEntry

app = FastAPI(
    title="MITRA — Mental Wellness Support Conversation Agent",
    description=(
        "An agentic AI companion for everyday mental wellness support. "
        "Not a medical device. Not a substitute for professional care."
    ),
    version=settings.version,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

orchestrator = Orchestrator()
UI_FILE = Path(__file__).resolve().parent.parent / "ui" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if UI_FILE.exists():
        return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>MITRA API</h1><p>See <a href='/docs'>/docs</a>.</p>")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **settings.as_dict()}


@app.post("/chat", response_model=AgentResponse)
def chat(req: ChatRequest) -> AgentResponse:
    if not (req.message or "").strip():
        raise HTTPException(400, "message must not be empty")
    if len(req.message) > 4000:
        raise HTTPException(413, "message too long")
    return orchestrator.handle(req.message, req.session_id, req.consent)


@app.post("/session")
def new_session(consent: bool = True) -> dict:
    return {"session_id": orchestrator.store.new_session(consent)}


@app.get("/session/{sid}/report")
def session_report(sid: str) -> JSONResponse:
    report = orchestrator.session_report(sid)
    if not report["session"]:
        raise HTTPException(404, "unknown session")
    return JSONResponse(report)


@app.post("/mood")
def log_mood(entry: MoodEntry) -> dict:
    rid = orchestrator.store.log_mood(entry.session_id, entry.score,
                                      entry.label, entry.note)
    return {"id": rid, "ok": True}


@app.get("/resources")
def resources(region: str = settings.region) -> dict:
    return {"region": region, "helplines": safety.helplines(region)}


@app.delete("/session/{sid}")
def wipe(sid: str) -> dict:
    """Right to erasure — removes every trace of a session."""
    orchestrator.store.wipe_session(sid)
    return {"deleted": sid}


@app.get("/tools")
def tools() -> dict:
    from . import tools as toolkit
    return {"tools": toolkit.specs()}
