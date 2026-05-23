from __future__ import annotations

import shutil
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from data_preparation.call_graph import build_call_graph
from data_preparation.download_repo import detect_repository_language, download_github_repository
from data_preparation.graph_storage import DEFAULT_GRAPH_PATH, persist_call_graph
from data_preparation.symbol_embeddings import DEFAULT_COLLECTION_NAME, persist_symbol_embeddings
from data_preparation.symbol_parser import Symbol, extract_repository_symbols

if TYPE_CHECKING:
    import chromadb
    import networkx as nx


DEFAULT_REPOSITORY_CACHE_DIR = "repositories"
DEFAULT_CHROMA_PATH = "chroma"


@dataclass(frozen=True)
class IndexDataResult:
    repository_path: Path
    language: Optional[str]
    symbols: list[Symbol]
    call_graph: "nx.DiGraph"
    embeddings_collection: "chromadb.Collection"
    graph_path: Path
    chroma_path: Path


def index_data(
    source,
    repository_cache_dir=DEFAULT_REPOSITORY_CACHE_DIR,
    chroma_path=DEFAULT_CHROMA_PATH,
    graph_path=DEFAULT_GRAPH_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_function=None,
    force_download: bool = False,
    batch_size: int = 100,
) -> IndexDataResult:
    """
    Index a local repository path or GitHub repository URL.

    The pipeline resolves the repository, extracts symbols, builds a NetworkX call
    graph, and persists symbol summary/code/comment embeddings to local ChromaDB.
    """
    repository_path, language = _resolve_repository(
        source=source,
        repository_cache_dir=repository_cache_dir,
        force_download=force_download,
    )
    symbols = extract_repository_symbols(repository_path)
    call_graph = build_call_graph(symbols)
    persisted_graph_path = persist_call_graph(call_graph, graph_path)
    embeddings_collection = persist_symbol_embeddings(
        repository_dir=repository_path,
        symbols=symbols,
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_function=embedding_function,
        batch_size=batch_size,
    )

    return IndexDataResult(
        repository_path=repository_path,
        language=language,
        symbols=symbols,
        call_graph=call_graph,
        embeddings_collection=embeddings_collection,
        graph_path=persisted_graph_path,
        chroma_path=Path(chroma_path).expanduser().resolve(),
    )


def _resolve_repository(source, repository_cache_dir, force_download: bool) -> tuple[Path, Optional[str]]:
    source_text = str(source).strip()
    if _is_github_url(source_text):
        repository_path = _repository_destination(source_text, repository_cache_dir)
        if force_download and repository_path.exists():
            _remove_directory(repository_path, Path(repository_cache_dir).expanduser().resolve())

        if not repository_path.exists() or not any(repository_path.iterdir()):
            language = download_github_repository(source_text, repository_path)
        else:
            language = detect_repository_language(repository_path)

        return repository_path.resolve(), language

    repository_path = Path(source).expanduser().resolve()
    if not repository_path.is_dir():
        raise NotADirectoryError(f"Repository source is not a directory or GitHub URL: {source}")

    return repository_path, detect_repository_language(repository_path)


def _is_github_url(source: str) -> bool:
    parsed = urllib.parse.urlparse(source)
    return (
        (parsed.scheme in {"http", "https", "ssh"} and parsed.hostname == "github.com")
        or source.startswith("git@github.com:")
    )


def _repository_destination(github_url: str, repository_cache_dir) -> Path:
    owner, repo = _github_owner_and_repo(github_url)
    safe_name = f"{owner}-{repo}".replace(".", "-")
    return Path(repository_cache_dir).expanduser().resolve() / safe_name


def _github_owner_and_repo(github_url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(github_url)

    if github_url.startswith("git@github.com:"):
        path = github_url[len("git@github.com:") :]
    else:
        path = parsed.path

    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("Expected a GitHub repository URL, for example https://github.com/owner/repo")

    owner = parts[0]
    repo = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return owner, repo


def _remove_directory(path: Path, allowed_parent: Path) -> None:
    resolved_path = path.resolve()
    resolved_parent = allowed_parent.resolve()
    if not _is_relative_to(resolved_path, resolved_parent):
        raise ValueError(f"Refusing to remove path outside repository cache: {resolved_path}")

    shutil.rmtree(resolved_path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False

    return True
