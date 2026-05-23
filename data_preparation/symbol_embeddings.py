from __future__ import annotations

import hashlib
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING

from data_preparation.symbol_parser import Symbol

if TYPE_CHECKING:
    import chromadb


DEFAULT_COLLECTION_NAME = "symbol_embeddings"


@dataclass(frozen=True)
class SymbolEmbeddingDocument:
    id: str
    document: str
    metadata: dict


def persist_symbol_embeddings(
    repository_dir,
    symbols: Iterable[Symbol],
    chroma_path="chroma",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_function=None,
    batch_size: int = 100,
):
    """
    Build symbol embedding documents and persist them to local ChromaDB storage.

    Chroma computes embeddings from the supplied documents using the collection's
    embedding function. Pass embedding_function to use a specific model/provider.
    """
    records = build_symbol_embedding_documents(repository_dir, symbols)
    client = create_chroma_client(chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"source": "repository-symbols"},
    )

    for batch in _batched(records, batch_size):
        collection.upsert(
            ids=[record.id for record in batch],
            documents=[record.document for record in batch],
            metadatas=[record.metadata for record in batch],
        )

    return collection


def persist_repository_symbol_embeddings(
    repository_dir,
    chroma_path="chroma",
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_function=None,
    batch_size: int = 100,
):
    from data_preparation.symbol_parser import extract_repository_symbols

    symbols = extract_repository_symbols(repository_dir)
    return persist_symbol_embeddings(
        repository_dir=repository_dir,
        symbols=symbols,
        chroma_path=chroma_path,
        collection_name=collection_name,
        embedding_function=embedding_function,
        batch_size=batch_size,
    )


def build_symbol_embedding_documents(
    repository_dir,
    symbols: Iterable[Symbol],
) -> List[SymbolEmbeddingDocument]:
    root = Path(repository_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository directory does not exist: {root}")

    records = []
    for symbol in symbols:
        code_body = _read_symbol_code(root, symbol)
        pieces = [
            ("summary", _symbol_summary(symbol)),
            ("code_body", code_body),
            ("comments", _extract_comments(symbol.file, code_body)),
        ]

        for piece_type, document in pieces:
            normalized_document = document.strip()
            if not normalized_document:
                continue

            records.append(
                SymbolEmbeddingDocument(
                    id=_document_id(symbol, piece_type),
                    document=normalized_document,
                    metadata=_metadata(symbol, piece_type),
                )
            )

    return _deduplicate_records(records)


def create_chroma_client(chroma_path) -> "chromadb.PersistentClient":
    chromadb = _load_chromadb()
    path = Path(chroma_path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _symbol_summary(symbol: Symbol) -> str:
    return (
        f"{symbol.type} symbol {symbol.name} in {symbol.file} "
        f"from line {symbol.start_line} to line {symbol.end_line}."
    )


def _read_symbol_code(repository_dir: Path, symbol: Symbol) -> str:
    file_path = (repository_dir / symbol.file).resolve()
    if not _is_relative_to(file_path, repository_dir) or not file_path.is_file():
        return ""

    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""

    start_index = max(symbol.start_line - 1, 0)
    end_index = min(symbol.end_line, len(lines))
    return "\n".join(lines[start_index:end_index])


def _extract_comments(file_name: str, code_body: str) -> str:
    if not code_body:
        return ""

    if file_name.endswith(".py"):
        return _extract_python_comments(code_body)

    if file_name.endswith(".java"):
        return _extract_java_comments(code_body)

    return ""


def _extract_python_comments(code_body: str) -> str:
    comments = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code_body).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                comments.append(token.string)
    except tokenize.TokenError:
        return "\n".join(_fallback_hash_comments(code_body))

    return "\n".join(comments)


def _extract_java_comments(code_body: str) -> str:
    comment_pattern = re.compile(r"//.*?$|/\*.*?\*/", flags=re.MULTILINE | re.DOTALL)
    return "\n".join(match.group(0) for match in comment_pattern.finditer(code_body))


def _fallback_hash_comments(code_body: str) -> List[str]:
    return [line.strip() for line in code_body.splitlines() if line.strip().startswith("#")]


def _metadata(symbol: Symbol, piece_type: str) -> dict:
    return {
        "symbol_name": symbol.name,
        "symbol_type": symbol.type,
        "piece_type": piece_type,
        "file": symbol.file,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
    }


def _document_id(symbol: Symbol, piece_type: str) -> str:
    raw_id = "|".join(
        [
            symbol.file,
            str(symbol.start_line),
            str(symbol.end_line),
            symbol.type,
            symbol.name,
            piece_type,
        ]
    )
    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
    return f"symbol-{digest}-{piece_type}"


def _deduplicate_records(records: List[SymbolEmbeddingDocument]) -> List[SymbolEmbeddingDocument]:
    unique_records = {}
    for record in records:
        unique_records.setdefault(record.id, record)

    return list(unique_records.values())


def _batched(records: List[SymbolEmbeddingDocument], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    for index in range(0, len(records), batch_size):
        yield records[index : index + batch_size]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False

    return True


def _load_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Missing ChromaDB dependency. Install project dependencies before persisting embeddings."
        ) from exc

    return chromadb
