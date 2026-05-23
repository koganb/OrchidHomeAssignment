from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient

import api

logger = logging.getLogger(__name__)


def test_process_and_chat_python_helloworld_repo(tmp_path: Path, monkeypatch) -> None:
    github_url = "https://github.com/dbarnett/python-helloworld"
    question = "Explain the main code flow in this repository."

    processed_root = tmp_path / "processed_repositories"
    chroma_root = tmp_path / "chroma"
    graph_root = tmp_path / "graphs"

    monkeypatch.setattr(api, "PROCESSED_ROOT", processed_root)
    monkeypatch.setattr(api, "CHROMA_ROOT", chroma_root)
    monkeypatch.setattr(api, "GRAPH_ROOT", graph_root)

    client = TestClient(api.app)

    process_response = client.post(
        "/process",
        json={"github_url": github_url, "force": True},
    )
    assert process_response.status_code == 200, process_response.text
    process_payload = process_response.json()
    assert Path(process_payload["repository_path"]).is_dir()
    assert process_payload["language"] == "Python"
    assert Path(process_payload["chroma_path"]).is_dir()
    assert Path(process_payload["networkx_path"]).is_file()
    assert process_payload["already_processed"] is False
    assert process_payload["symbol_count"] > 0

    manifest_path = processed_root / "dbarnett-python-helloworld" / "manifest.json"
    assert manifest_path.is_file()

    chat_response = client.post(
        "/chat",
        json={
            "question": question,
            "chroma_path": process_payload["chroma_path"],
            "networkx_path": process_payload["networkx_path"],
        },
    )
    assert chat_response.status_code == 200, chat_response.text
    answer = chat_response.json()["answer"]
    logger.info(f"Chat response: {answer}")
    assert isinstance(answer, str)
    assert answer.strip()


def test_process_start_returns_background_job_status(tmp_path: Path, monkeypatch) -> None:
    processed_root = tmp_path / "processed_repositories"
    chroma_root = tmp_path / "chroma"
    graph_root = tmp_path / "graphs"

    monkeypatch.setattr(api, "PROCESSED_ROOT", processed_root)
    monkeypatch.setattr(api, "CHROMA_ROOT", chroma_root)
    monkeypatch.setattr(api, "GRAPH_ROOT", graph_root)
    monkeypatch.setattr(
        api,
        "_process_repository_now",
        lambda request: api.ProcessResponse(
            repository_path=str(processed_root / "owner-repo"),
            language="Python",
            chroma_path=str(chroma_root / "owner-repo"),
            networkx_path=str(graph_root / "owner-repo.pkl"),
            already_processed=False,
            symbol_count=1,
        ),
    )

    client = TestClient(api.app)
    start_response = client.post(
        "/process/start",
        json={"github_url": "https://github.com/owner/repo"},
    )

    assert start_response.status_code == 200, start_response.text
    job_id = start_response.json()["job_id"]

    status_response = client.get(f"/process/status/{job_id}")
    assert status_response.status_code == 200, status_response.text

    payload = status_response.json()
    assert payload["status"] == "completed"
    assert payload["result"]["symbol_count"] == 1


def test_chat_rejects_non_repository_question(monkeypatch) -> None:
    def fail_if_called(**kwargs):
        raise AssertionError("run_react_agent should not be called for rejected questions")

    monkeypatch.setattr(api, "run_react_agent", fail_if_called)
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={
            "question": "What is the capital of France?",
            "chroma_path": "unused",
            "networkx_path": "unused",
        },
    )

    assert response.status_code == 400
    assert "repository code" in response.json()["detail"]


def test_chat_accepts_repository_code_question(monkeypatch) -> None:
    monkeypatch.setattr(api, "run_react_agent", lambda **kwargs: "Repository code explanation.")
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={
            "question": "Explain the process_repository code flow.",
            "chroma_path": "unused",
            "networkx_path": "unused",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "Repository code explanation."
