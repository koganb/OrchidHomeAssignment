from __future__ import annotations

import re


REJECTION_MESSAGE = (
    "Chat only supports questions that ask to explain repository code, execution flow, "
    "call relationships, classes, methods, functions, modules, or implementation details."
)

_CODE_OR_FLOW_TERMS = {
    "architecture",
    "call",
    "calls",
    "class",
    "classes",
    "code",
    "component",
    "components",
    "dependency",
    "dependencies",
    "explain",
    "file",
    "flow",
    "function",
    "functions",
    "implementation",
    "implements",
    "logic",
    "method",
    "methods",
    "module",
    "modules",
    "package",
    "path",
    "pipeline",
    "repository",
    "repo",
    "service",
    "trace",
}

_NON_REPOSITORY_TERMS = {
    "capital",
    "career",
    "cook",
    "dinner",
    "email",
    "essay",
    "finance",
    "movie",
    "news",
    "poem",
    "recipe",
    "song",
    "stock",
    "translate",
    "travel",
    "weather",
}


def validate_repository_chat_question(question: str) -> None:
    if not is_repository_chat_question(question):
        raise ValueError(REJECTION_MESSAGE)


def is_repository_chat_question(question: str) -> bool:
    normalized = question.strip().lower()
    if not normalized:
        return False

    terms = set(re.findall(r"[a-z][a-z0-9_]*", normalized))
    if terms & _NON_REPOSITORY_TERMS and not terms & _CODE_OR_FLOW_TERMS:
        return False

    if terms & _CODE_OR_FLOW_TERMS:
        return True

    return _looks_like_code_reference(question)


def _looks_like_code_reference(question: str) -> bool:
    return bool(
        re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", question)
        or re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b", question)
        or re.search(r"\b[\w.-]+\.(py|java)\b", question, flags=re.IGNORECASE)
    )
