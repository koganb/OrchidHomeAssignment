from __future__ import annotations

import pickle
from pathlib import Path


DEFAULT_GRAPH_PATH = "graphs/call_graph.pkl"


def persist_call_graph(call_graph, graph_path=DEFAULT_GRAPH_PATH) -> Path:
    path = Path(graph_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as graph_file:
        pickle.dump(call_graph, graph_file)

    return path


def load_call_graph(graph_path):
    path = Path(graph_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NetworkX graph file does not exist: {path}")

    with path.open("rb") as graph_file:
        return pickle.load(graph_file)
