from __future__ import annotations

import re
from pathlib import Path
from langchain.tools import tool
import chromadb

from data_preparation.graph_storage import load_call_graph


def create_chat_tools(chroma_path: str, networkx_path: str, collection_name: str):
    @tool
    def explain_feature(query: str) -> str:
        """Use for questions about what a feature, class, method, or code area does."""
        return _explain_feature(query, chroma_path, collection_name)

    @tool
    def trace_flow(query: str) -> str:
        """Use for questions about execution flow, call chains, dependencies, or what calls what."""
        return _trace_flow(query, networkx_path)

    return [explain_feature, trace_flow]


def _explain_feature(query: str, chroma_path: str, collection_name: str) -> str:
    client = chromadb.PersistentClient(path=str(Path(chroma_path).expanduser().resolve()))
    collection = client.get_collection(name=collection_name)
    results = collection.query(query_texts=[query], n_results=8)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "No relevant embedded code context was found."

    chunks = []
    for document, metadata in zip(documents, metadatas):
        chunks.append(
            "\n".join(
                [
                    f"Symbol: {metadata.get('symbol_name')}",
                    f"Type: {metadata.get('symbol_type')} / {metadata.get('piece_type')}",
                    f"File: {metadata.get('file')}:{metadata.get('start_line')}",
                    "Content:",
                    document,
                ]
            )
        )

    return "\n\n---\n\n".join(chunks)


def _trace_flow(query: str, networkx_path: str) -> str:
    graph = load_call_graph(networkx_path)
    terms = _query_terms(query)
    matching_nodes = [
        node
        for node in graph.nodes
        if any(term in node.name.lower() or term in node.file.lower() for term in terms)
    ]
    if not matching_nodes:
        return "No matching methods or calls were found in the call graph."

    lines = []
    for node in matching_nodes[:8]:
        lines.append(f"{node.name} ({node.type}) in {node.file}:{node.start_line}-{node.end_line}")
        outgoing = list(graph.out_edges(node, data=True))[:10]
        incoming = list(graph.in_edges(node, data=True))[:10]

        if incoming:
            lines.append("Called by:")
            for caller, _, data in incoming:
                lines.append(
                    f"- {caller.name} via {data.get('call_name')} "
                    f"at {data.get('file')}:{data.get('start_line')}"
                )

        if outgoing:
            lines.append("Calls:")
            for _, callee, data in outgoing:
                lines.append(
                    f"- {callee.name} via {data.get('call_name')} "
                    f"at {data.get('file')}:{data.get('start_line')}"
                )

        lines.append("")

    return "\n".join(lines).strip()


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query.lower())
    stop_words = {
        "and",
        "are",
        "call",
        "calls",
        "does",
        "flow",
        "for",
        "from",
        "how",
        "the",
        "this",
        "what",
        "where",
        "which",
    }
    return [term for term in terms if term not in stop_words]
