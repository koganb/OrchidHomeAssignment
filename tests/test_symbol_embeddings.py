from __future__ import annotations

from pathlib import Path

from data_preparation.symbol_embeddings import build_symbol_embedding_documents
from data_preparation.symbol_parser import Symbol


def test_build_symbol_embedding_documents_deduplicates_record_ids(tmp_path: Path) -> None:
    source_file = tmp_path / "example.py"
    source_file.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

    symbol = Symbol(
        name="greet",
        type="function",
        file="example.py",
        start_line=1,
        end_line=2,
    )

    records = build_symbol_embedding_documents(tmp_path, [symbol, symbol])

    ids = [record.id for record in records]
    assert len(ids) == len(set(ids))
    assert len(ids) == 2
