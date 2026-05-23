from __future__ import annotations

import requests
import time

from streamlit_gui.models import ProcessedRepository


REQUEST_TIMEOUT_SECONDS = 30
HEALTH_TIMEOUT_SECONDS = 3
PROCESS_POLL_INTERVAL_SECONDS = 2
PROCESS_MAX_WAIT_SECONDS = 1800


class ApiClientError(RuntimeError):
    pass


def process_repository(api_base_url: str, github_url: str) -> ProcessedRepository:
    payload = {"github_url": github_url}
    job = _post(f"{api_base_url}/process/start", payload)
    response = _wait_for_process_job(api_base_url, job["job_id"])
    return ProcessedRepository(github_url=github_url, **response)


def api_is_available(api_base_url: str) -> bool:
    try:
        response = requests.get(f"{api_base_url}/health", timeout=HEALTH_TIMEOUT_SECONDS)
        return response.status_code == 200
    except requests.RequestException:
        return False


def ask_question(
    api_base_url: str,
    question: str,
    chroma_path: str,
    networkx_path: str,
    collection_name: str = "symbol_embeddings",
) -> str:
    payload = {
        "question": question,
        "chroma_path": chroma_path,
        "networkx_path": networkx_path,
        "collection_name": collection_name,
    }
    response = _post(f"{api_base_url}/chat", payload)
    return response["answer"]


def _post(url: str, payload: dict) -> dict:
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise ApiClientError("The API request timed out.") from exc
    except requests.ConnectionError as exc:
        raise ApiClientError(
            f"Could not connect to the API server at {url}. Start FastAPI first or set REPOSITORY_API_BASE_URL."
        ) from exc
    except requests.HTTPError as exc:
        raise ApiClientError(_error_detail(response)) from exc

    return response.json()


def _get(url: str) -> dict:
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise ApiClientError("The API request timed out.") from exc
    except requests.ConnectionError as exc:
        raise ApiClientError(
            f"Could not connect to the API server at {url}. Start FastAPI first or set REPOSITORY_API_BASE_URL."
        ) from exc
    except requests.HTTPError as exc:
        raise ApiClientError(_error_detail(response)) from exc

    return response.json()


def _wait_for_process_job(api_base_url: str, job_id: str) -> dict:
    deadline = time.monotonic() + PROCESS_MAX_WAIT_SECONDS

    while time.monotonic() < deadline:
        payload = _get(f"{api_base_url}/process/status/{job_id}")
        status = payload["status"]

        if status == "completed":
            return payload["result"]

        if status == "failed":
            raise ApiClientError(payload.get("error") or "Repository processing failed.")

        time.sleep(PROCESS_POLL_INTERVAL_SECONDS)

    raise ApiClientError("Repository processing did not finish before the maximum wait time.")


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"API request failed with status {response.status_code}."

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail

    return f"API request failed with status {response.status_code}."
