from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".idea",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}


@dataclass(frozen=True)
class Symbol:
    name: str
    type: str
    file: str
    start_line: int
    end_line: int


def extract_repository_symbols(repository_dir) -> List[Symbol]:
    """
    Extract symbols from supported source files in a downloaded repository.

    Only Python and Java files are processed. Parsing is done with Tree-sitter;
    install the dependencies declared in pyproject.toml before calling this
    function.
    """
    root = Path(repository_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository directory does not exist: {root}")

    symbols = []
    for file_path in _iter_supported_source_files(root):
        if file_path.suffix == ".py":
            symbols.extend(_extract_python_symbols(file_path, root))
        elif file_path.suffix == ".java":
            symbols.extend(_extract_java_symbols(file_path, root))

    return symbols


def _iter_supported_source_files(repository_dir: Path) -> Iterable[Path]:
    for file_path in repository_dir.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix in {".py", ".java"}
            and not _is_in_excluded_dir(file_path, repository_dir)
        ):
            yield file_path


def _extract_python_symbols(file_path: Path, repository_dir: Path) -> List[Symbol]:
    source, root_node = _parse_source_file(file_path, "python")
    relative_file = _relative_file(file_path, repository_dir)
    symbols = []

    def walk(node, class_stack):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = _node_text(source, name_node)
            if class_name:
                qualified_name = ".".join(class_stack + [class_name])
                symbols.append(_symbol(qualified_name, "class", relative_file, node))
                _add_python_docstring_symbol(symbols, source, node, relative_file, qualified_name)
                next_stack = class_stack + [class_name]
            else:
                next_stack = class_stack

            for child in node.children:
                walk(child, next_stack)
            return

        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            method_name = _node_text(source, name_node)
            if method_name:
                qualified_name = ".".join(class_stack + [method_name])
                symbol_type = "method" if class_stack else "function"
                symbols.append(_symbol(qualified_name, symbol_type, relative_file, node))
                _add_python_docstring_symbol(symbols, source, node, relative_file, qualified_name)

        elif node.type in {"import_statement", "import_from_statement"}:
            import_name = _single_line_node_text(source, node)
            symbols.append(_symbol(import_name, "import", relative_file, node))

        elif node.type == "call":
            call_name = _python_call_name(source, node)
            if call_name:
                symbols.append(_symbol(call_name, "call", relative_file, node))

        for child in node.children:
            walk(child, class_stack)

    walk(root_node, [])
    return symbols


def _extract_java_symbols(file_path: Path, repository_dir: Path) -> List[Symbol]:
    source, root_node = _parse_source_file(file_path, "java")
    relative_file = _relative_file(file_path, repository_dir)
    symbols = []

    def walk(node, class_stack):
        if node.type in {"class_declaration", "interface_declaration", "enum_declaration"}:
            name_node = node.child_by_field_name("name")
            class_name = _node_text(source, name_node)
            if class_name:
                qualified_name = ".".join(class_stack + [class_name])
                symbols.append(_symbol(qualified_name, "class", relative_file, node))
                _add_java_docstring_symbol(symbols, source, node, relative_file, qualified_name)
                next_stack = class_stack + [class_name]
            else:
                next_stack = class_stack

            for child in node.children:
                walk(child, next_stack)
            return

        if node.type in {"method_declaration", "constructor_declaration"}:
            name_node = node.child_by_field_name("name")
            method_name = _node_text(source, name_node)
            if method_name:
                qualified_name = ".".join(class_stack + [method_name])
                symbols.append(_symbol(qualified_name, "method", relative_file, node))
                _add_java_docstring_symbol(symbols, source, node, relative_file, qualified_name)

        elif node.type == "import_declaration":
            import_name = _single_line_node_text(source, node)
            symbols.append(_symbol(import_name, "import", relative_file, node))

        elif node.type in {"method_invocation", "object_creation_expression"}:
            call_name = _java_call_name(source, node)
            if call_name:
                symbols.append(_symbol(call_name, "call", relative_file, node))

        for child in node.children:
            walk(child, class_stack)

    walk(root_node, [])
    return symbols


def _parse_source_file(file_path: Path, language_name: str):
    from tree_sitter import Language, Parser

    language = _load_tree_sitter_language(Language, language_name)
    parser = Parser()

    if hasattr(parser, "set_language"):
        parser.set_language(language)
    else:
        parser.language = language

    source = file_path.read_bytes()
    tree = parser.parse(source)
    return source, tree.root_node


def _load_tree_sitter_language(language_class, language_name: str):
    try:
        if language_name == "python":
            import tree_sitter_python as grammar
        elif language_name == "java":
            import tree_sitter_java as grammar
        else:
            raise ValueError(f"Unsupported Tree-sitter language: {language_name}")
    except ImportError as exc:
        raise RuntimeError(
            "Missing Tree-sitter grammar packages. Install project dependencies first."
        ) from exc

    grammar_language = grammar.language()
    try:
        return language_class(grammar_language)
    except TypeError:
        return grammar_language


def _add_python_docstring_symbol(
    symbols: List[Symbol],
    source: bytes,
    declaration_node,
    relative_file: str,
    owner_name: str,
) -> None:
    body = declaration_node.child_by_field_name("body")
    if body is None:
        return

    first_statement = _first_named_child(body)
    if first_statement is None or first_statement.type != "expression_statement":
        return

    string_node = _first_named_child(first_statement)
    if string_node is not None and string_node.type == "string":
        symbols.append(_symbol(f"{owner_name}.__doc__", "docstring", relative_file, string_node))


def _add_java_docstring_symbol(
    symbols: List[Symbol],
    source: bytes,
    declaration_node,
    relative_file: str,
    owner_name: str,
) -> None:
    previous = declaration_node.prev_named_sibling
    if previous is not None and previous.type == "block_comment":
        text = _node_text(source, previous).strip()
        if text.startswith("/**"):
            symbols.append(_symbol(f"{owner_name}.__doc__", "docstring", relative_file, previous))


def _python_call_name(source: bytes, call_node) -> Optional[str]:
    function_node = call_node.child_by_field_name("function")
    return _expression_name(source, function_node)


def _java_call_name(source: bytes, call_node) -> Optional[str]:
    if call_node.type == "object_creation_expression":
        type_node = call_node.child_by_field_name("type")
        name = _expression_name(source, type_node)
        return f"new {name}" if name else None

    object_node = call_node.child_by_field_name("object")
    name_node = call_node.child_by_field_name("name")
    method_name = _node_text(source, name_node)
    object_name = _expression_name(source, object_node)

    if object_name and method_name:
        return f"{object_name}.{method_name}"

    return method_name


def _expression_name(source: bytes, node) -> Optional[str]:
    if node is None:
        return None

    if node.type in {"identifier", "type_identifier", "property_identifier"}:
        return _node_text(source, node)

    if node.type in {"attribute", "field_access", "scoped_identifier"}:
        return _single_line_node_text(source, node)

    return _single_line_node_text(source, node)


def _first_named_child(node):
    for child in node.named_children:
        return child
    return None


def _symbol(name: str, symbol_type: str, relative_file: str, node) -> Symbol:
    return Symbol(
        name=name,
        type=symbol_type,
        file=relative_file,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
    )


def _relative_file(file_path: Path, repository_dir: Path) -> str:
    return file_path.relative_to(repository_dir).as_posix()


def _single_line_node_text(source: bytes, node) -> str:
    return " ".join(_node_text(source, node).split())


def _node_text(source: bytes, node) -> str:
    if node is None:
        return ""

    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _is_in_excluded_dir(file_path: Path, repository_dir: Path) -> bool:
    relative_parts = file_path.relative_to(repository_dir).parts
    return any(part in EXCLUDED_DIRS for part in relative_parts[:-1])
