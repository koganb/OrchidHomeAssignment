from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Iterable, List, Optional, TYPE_CHECKING

from data_preparation.symbol_parser import Symbol

if TYPE_CHECKING:
    import networkx as nx


def build_call_graph(symbols: Iterable[Symbol]) -> "nx.DiGraph":
    """
    Build a directed call graph from extracted repository symbols.

    Method symbols are graph roots. Each call expression found inside a method's
    source range creates an edge from the enclosing method to either a resolved
    method symbol or to the original unresolved call symbol.
    """
    nx = _load_networkx()
    symbol_list = list(symbols)
    methods = [symbol for symbol in symbol_list if symbol.type == "method"]
    calls = [symbol for symbol in symbol_list if symbol.type == "call"]
    resolver = _MethodResolver(methods)

    graph = nx.DiGraph()
    for method in methods:
        graph.add_node(method, **_node_attributes(method, kind="method", resolved=True))

    for call in calls:
        enclosing_method = _find_enclosing_method(call, methods)
        if enclosing_method is None:
            continue

        target = resolver.resolve(call.name, enclosing_method)
        if target is None:
            target = call
            graph.add_node(target, **_node_attributes(call, kind="call", resolved=False))
        else:
            graph.add_node(target, **_node_attributes(target, kind="method", resolved=True))

        _add_call_edge(graph, enclosing_method, target, call)

    return graph


def build_call_graph_from_repository(repository_dir) -> "nx.DiGraph":
    from data_preparation.symbol_parser import extract_repository_symbols

    return build_call_graph(extract_repository_symbols(repository_dir))


class _MethodResolver:
    def __init__(self, methods: List[Symbol]) -> None:
        self._by_exact_name: DefaultDict[str, List[Symbol]] = defaultdict(list)
        self._by_file_and_short_name: DefaultDict[tuple[str, str], List[Symbol]] = defaultdict(list)
        self._by_short_name: DefaultDict[str, List[Symbol]] = defaultdict(list)

        for method in methods:
            self._by_exact_name[method.name].append(method)
            short_name = method.name.rsplit(".", 1)[-1]
            self._by_file_and_short_name[(method.file, short_name)].append(method)
            self._by_short_name[short_name].append(method)

    def resolve(self, call_name: str, caller: Symbol) -> Optional[Symbol]:
        exact_matches = self._by_exact_name.get(call_name, [])
        if len(exact_matches) == 1:
            return exact_matches[0]

        normalized_name = _normalize_call_name(call_name)
        caller_class = _class_name(caller.name)

        if caller_class:
            same_class_name = f"{caller_class}.{normalized_name}"
            same_class_matches = self._by_exact_name.get(same_class_name, [])
            if len(same_class_matches) == 1:
                return same_class_matches[0]

        same_file_matches = self._by_file_and_short_name.get((caller.file, normalized_name), [])
        if len(same_file_matches) == 1:
            return same_file_matches[0]

        global_matches = self._by_short_name.get(normalized_name, [])
        if len(global_matches) == 1:
            return global_matches[0]

        return None


def _find_enclosing_method(call: Symbol, methods: List[Symbol]) -> Optional[Symbol]:
    matching_methods = [
        method
        for method in methods
        if method.file == call.file
        and method.start_line <= call.start_line
        and call.end_line <= method.end_line
    ]
    if not matching_methods:
        return None

    return min(matching_methods, key=lambda method: method.end_line - method.start_line)


def _add_call_edge(graph, caller: Symbol, target: Symbol, call: Symbol) -> None:
    call_site = {
        "call_name": call.name,
        "file": call.file,
        "start_line": call.start_line,
        "end_line": call.end_line,
    }

    if graph.has_edge(caller, target):
        edge = graph.edges[caller, target]
        edge["count"] += 1
        edge["call_sites"].append(call_site)
        return

    graph.add_edge(
        caller,
        target,
        call_name=call.name,
        file=call.file,
        start_line=call.start_line,
        end_line=call.end_line,
        resolved=target.type == "method",
        count=1,
        call_sites=[call_site],
    )


def _normalize_call_name(call_name: str) -> str:
    if call_name.startswith("new "):
        call_name = call_name[len("new ") :]

    for prefix in ("self.", "this."):
        if call_name.startswith(prefix):
            call_name = call_name[len(prefix) :]

    return call_name.rsplit(".", 1)[-1]


def _class_name(method_name: str) -> Optional[str]:
    parts = method_name.rsplit(".", 1)
    if len(parts) != 2:
        return None

    return parts[0]


def _node_attributes(symbol: Symbol, kind: str, resolved: bool) -> dict:
    return {
        "name": symbol.name,
        "type": symbol.type,
        "kind": kind,
        "file": symbol.file,
        "start_line": symbol.start_line,
        "end_line": symbol.end_line,
        "resolved": resolved,
    }


def _load_networkx():
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "Missing NetworkX dependency. Install project dependencies before building a call graph."
        ) from exc

    return nx
