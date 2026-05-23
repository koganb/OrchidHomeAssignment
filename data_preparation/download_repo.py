from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

from data_preparation.symbol_parser import Symbol, extract_repository_symbols


EXTENSION_TO_LANGUAGE = {
    ".c": "C",
    ".cc": "C++",
    ".clj": "Clojure",
    ".cljs": "Clojure",
    ".cpp": "C++",
    ".cs": "C#",
    ".css": "CSS",
    ".dart": "Dart",
    ".ex": "Elixir",
    ".exs": "Elixir",
    ".fs": "F#",
    ".go": "Go",
    ".h": "C",
    ".hpp": "C++",
    ".hs": "Haskell",
    ".html": "HTML",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".m": "Objective-C",
    ".mm": "Objective-C++",
    ".php": "PHP",
    ".pl": "Perl",
    ".pm": "Perl",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".scala": "Scala",
    ".sh": "Shell",
    ".swift": "Swift",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".vue": "Vue",
}

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


def download_github_repository(github_url: str, destination_dir) -> Optional[str]:
    """
    Download a GitHub repository into destination_dir and return its primary language.

    The function expects git to be installed and available on PATH. The destination
    directory must not already contain files.

    Args:
        github_url: HTTPS or SSH GitHub repository URL.
        destination_dir: Directory where the repository should be cloned.

    Returns:
        The repository's primary language, or None if no language can be detected.

    Raises:
        ValueError: If github_url is not a supported GitHub repository URL.
        FileExistsError: If destination_dir already exists and is not empty.
        RuntimeError: If git is unavailable or the clone fails.
    """
    owner, repo = _parse_github_url(github_url)
    destination = Path(destination_dir).expanduser().resolve()
    _ensure_destination_is_empty(destination)

    _clone_repository(github_url, destination)

    github_language = _get_github_primary_language(owner, repo)
    if github_language:
        return github_language

    return _detect_language_from_files(destination)


def detect_repository_language(repository_dir) -> Optional[str]:
    repository_path = Path(repository_dir).expanduser().resolve()
    if not repository_path.is_dir():
        raise NotADirectoryError(f"Repository directory does not exist: {repository_path}")

    return _detect_language_from_files(repository_path)


def get_github_primary_language(github_url: str) -> Optional[str]:
    owner, repo = _parse_github_url(github_url)
    return _get_github_primary_language(owner, repo)


def _parse_github_url(github_url: str) -> Tuple[str, str]:
    parsed = urllib.parse.urlparse(github_url.strip())

    if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "github.com":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return parts[0], _strip_git_suffix(parts[1])

    if parsed.scheme == "ssh" and parsed.hostname == "github.com":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return parts[0], _strip_git_suffix(parts[1])

    if github_url.startswith("git@github.com:"):
        path = github_url[len("git@github.com:") :]
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 2:
            return parts[0], _strip_git_suffix(parts[1])

    raise ValueError("Expected a GitHub repository URL, for example https://github.com/owner/repo")


def _strip_git_suffix(repo: str) -> str:
    return repo[:-4] if repo.endswith(".git") else repo


def _ensure_destination_is_empty(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination directory is not empty: {destination}")

    destination.mkdir(parents=True, exist_ok=True)


def _clone_repository(github_url: str, destination: Path) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is not installed or is not available on PATH")

    result = subprocess.run(
        ["git", "clone", "--depth", "1", github_url, str(destination)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone repository: {result.stderr.strip()}")


def _get_github_primary_language(owner: str, repo: str) -> Optional[str]:
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "orchid-home-assignment",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    language = payload.get("language")
    return language if isinstance(language, str) and language else None


def _detect_language_from_files(repository_dir: Path) -> Optional[str]:
    language_sizes: Counter[str] = Counter()

    for file_path in repository_dir.rglob("*"):
        if not file_path.is_file() or _is_in_excluded_dir(file_path, repository_dir):
            continue

        language = EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower())
        if language is None:
            continue

        try:
            language_sizes[language] += file_path.stat().st_size
        except OSError:
            continue

    if not language_sizes:
        return None

    return language_sizes.most_common(1)[0][0]


def _is_in_excluded_dir(file_path: Path, repository_dir: Path) -> bool:
    relative_parts = file_path.relative_to(repository_dir).parts
    return any(part in EXCLUDED_DIRS for part in relative_parts[:-1])


if __name__ == "__main__":
    language = download_github_repository(
        "https://github.com/octocat/Hello-World",
        "downloaded-repository",
    )
    print(f"Repository language: {language}")
