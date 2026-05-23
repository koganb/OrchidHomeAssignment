from __future__ import annotations

import os


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def api_base_url() -> str:
    return os.getenv("REPOSITORY_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
