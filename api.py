from __future__ import annotations

import json
import re
import threading
import urllib.parse
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from chat.agent import run_react_agent
from chat.request_policy import REJECTION_MESSAGE, validate_repository_chat_question
from data_preparation.download_repo import get_github_primary_language
from data_preparation.index_data import index_data


SUPPORTED_LANGUAGES = {"Java", "Python"}
PROCESSED_ROOT = Path("processed_repositories").resolve()
CHROMA_ROOT = Path("chroma").resolve()
GRAPH_ROOT = Path("graphs").resolve()

app = FastAPI(title="Repository Intelligence API")
_jobs_lock = threading.Lock()
_process_jobs: dict[str, dict] = {}


class ProcessRequest(BaseModel):
    github_url: str = Field(..., description="GitHub repository URL")
    force: bool = Field(False, description="Reprocess even if cached output exists")


class ProcessResponse(BaseModel):
    repository_path: str
    language: str
    chroma_path: str
    networkx_path: str
    already_processed: bool
    symbol_count: int


class ProcessJobResponse(BaseModel):
    job_id: str
    status: str


class ProcessStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[ProcessResponse] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    question: str
    chroma_path: str
    networkx_path: str
    collection_name: str = "symbol_embeddings"


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
def process_repository(request: ProcessRequest) -> ProcessResponse:
    return _process_repository_now(request)


@app.post("/process/start", response_model=ProcessJobResponse)
def start_process_repository(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
) -> ProcessJobResponse:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _process_jobs[job_id] = {"status": "queued", "result": None, "error": None}

    background_tasks.add_task(_run_process_job, job_id, request)
    return ProcessJobResponse(job_id=job_id, status="queued")


@app.get("/process/status/{job_id}", response_model=ProcessStatusResponse)
def process_repository_status(job_id: str) -> ProcessStatusResponse:
    with _jobs_lock:
        job = _process_jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Processing job was not found.")

    result = job.get("result")
    return ProcessStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=ProcessResponse(**result) if result is not None else None,
        error=job.get("error"),
    )


def _process_repository_now(request: ProcessRequest) -> ProcessResponse:
    repo_key = _repo_key(request.github_url)
    manifest_path = _manifest_path(repo_key)

    if manifest_path.is_file() and not request.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ProcessResponse(**manifest, already_processed=True)

    language = _github_primary_language(request.github_url)

    if language is not None and language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported repository language: {language}. Supported languages: Java, Python.",
        )

    chroma_path = CHROMA_ROOT / repo_key
    graph_path = GRAPH_ROOT / f"{repo_key}.pkl"
    result = index_data(
        source=request.github_url,
        repository_cache_dir=PROCESSED_ROOT,
        chroma_path=chroma_path,
        graph_path=graph_path,
        force_download=request.force,
    )

    response_data = {
        "repository_path": str(result.repository_path),
        "language": result.language or language,
        "chroma_path": str(result.chroma_path),
        "networkx_path": str(result.graph_path),
        "symbol_count": len(result.symbols),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(response_data, indent=2), encoding="utf-8")

    return ProcessResponse(**response_data, already_processed=False)


def _run_process_job(job_id: str, request: ProcessRequest) -> None:
    _update_process_job(job_id, status="running")
    try:
        result = _process_repository_now(request)
    except Exception as exc:
        _update_process_job(job_id, status="failed", error=str(exc))
        return

    _update_process_job(job_id, status="completed", result=result.model_dump())


def _update_process_job(job_id: str, **changes) -> None:
    with _jobs_lock:
        job = _process_jobs.get(job_id)
        if job is None:
            return

        job.update(changes)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        validate_repository_chat_question(request.question)
    except ValueError:
        raise HTTPException(status_code=400, detail=REJECTION_MESSAGE) from None

    try:
        answer = run_react_agent(
            question=request.question,
            chroma_path=request.chroma_path,
            networkx_path=request.networkx_path,
            collection_name=request.collection_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ChatResponse(answer=answer)


def _github_primary_language(github_url: str) -> Optional[str]:
    return get_github_primary_language(github_url)


def _github_owner_and_repo(github_url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(github_url)
    if parsed.scheme not in {"http", "https", "ssh"} and not github_url.startswith("git@github.com:"):
        raise HTTPException(status_code=400, detail="Expected a GitHub repository URL")

    path = github_url[len("git@github.com:") :] if github_url.startswith("git@github.com:") else parsed.path
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Expected a GitHub repository URL")

    repo = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return parts[0], repo


def _repo_key(github_url: str) -> str:
    owner, repo = _github_owner_and_repo(github_url)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{owner}-{repo}").strip("-").lower()


def _manifest_path(repo_key: str) -> Path:
    return PROCESSED_ROOT / repo_key / "manifest.json"
