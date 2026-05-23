from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessedRepository:
    github_url: str
    repository_path: str
    language: str
    chroma_path: str
    networkx_path: str
    already_processed: bool
    symbol_count: int
