"""MCP server with the file tools the Coder uses (Day 4).

All tools work ONLY inside the target repo given by the TARGET_REPO env var.
Run (stdio):  uv run python -m code_review_agent.mcp_server
"""
import os
import re
from pathlib import Path

BLOCKED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
WRITABLE_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini"}
MAX_READ_CHARS = 20_000
MAX_WRITE_CHARS = 50_000
MAX_LIST = 200


# ----------------------------------------------------------------- safety helpers
def repo_root() -> Path:
    root = os.getenv("TARGET_REPO")
    if not root:
        raise RuntimeError("TARGET_REPO is not set (path of the repo the agent edits)")
    p = Path(root).resolve()
    if not p.is_dir():
        raise RuntimeError(f"TARGET_REPO is not a folder: {p}")
    return p


def _is_blocked(rel_parts: tuple[str, ...], name: str) -> bool:
    return bool(set(rel_parts) & BLOCKED_PARTS) or name.startswith(".env")


def _safe_path(path: str) -> Path:
    """Resolve `path` inside the repo, or raise ValueError."""
    root = repo_root()
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise ValueError("Path is outside the repo")
    if _is_blocked(p.relative_to(root).parts, p.name):
        raise ValueError("Path is not allowed (secrets, .git, .venv, caches)")
    return p


def _iter_files(root: Path, pattern: str = "*"):
    for p in sorted(root.rglob(pattern)):
        if p.is_file() and not _is_blocked(p.relative_to(root).parts, p.name):
            yield p


# --------------------------------------------------------------------------- tools
def list_files(subdir: str = ".") -> dict:
    """List files in the repo (relative paths). Skips .git, .venv, caches and .env files."""
    root = repo_root()
    base = _safe_path(subdir)
    if not base.is_dir():
        raise ValueError(f"Not a folder: {subdir}")
    files = [p.relative_to(root).as_posix() for p in _iter_files(base)]
    return {"files": files[:MAX_LIST], "truncated": len(files) > MAX_LIST}


def read_file(path: str) -> dict:
    """Read a text file from the repo."""
    p = _safe_path(path)
    if not p.is_file():
        raise ValueError(f"File not found: {path}")
    text = p.read_text(encoding="utf-8", errors="replace")
    return {
        "path": path,
        "content": text[:MAX_READ_CHARS],
        "truncated": len(text) > MAX_READ_CHARS,
    }


def write_file(path: str, content: str) -> dict:
    """Create or overwrite a text file in the repo (whole file content)."""
    p = _safe_path(path)
    if p.suffix.lower() not in WRITABLE_SUFFIXES:
        raise ValueError(f"Cannot write '{p.suffix}' files. Allowed: {sorted(WRITABLE_SUFFIXES)}")
    if len(content) > MAX_WRITE_CHARS:
        raise ValueError(f"Content too large (max {MAX_WRITE_CHARS} characters)")
    existed = p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return {"path": path, "bytes_written": len(content.encode("utf-8")), "created": not existed}


def search_code(query: str, glob: str = "*.py", regex: bool = False, max_results: int = 30) -> dict:
    """Search file contents. Plain case-insensitive text by default, or a regex."""
    root = repo_root()
    try:
        pattern = re.compile(query if regex else re.escape(query), re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Invalid regex: {e}")

    matches: list[dict] = []
    for p in _iter_files(root, glob):
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, start=1):
            if pattern.search(line):
                matches.append(
                    {"file": p.relative_to(root).as_posix(), "line": i, "text": line.strip()[:200]}
                )
                if len(matches) >= max_results:
                    return {"matches": matches, "truncated": True}
    return {"matches": matches, "truncated": False}


# -------------------------------------------------------------------------- server
TOOLS = (list_files, read_file, write_file, search_code)


def build_server():
    from fastmcp import FastMCP  # imported here so the tool functions are testable without it

    mcp = FastMCP("code-review-tools")
    for fn in TOOLS:
        mcp.tool(fn)
    return mcp


def main() -> None:
    build_server().run()  # stdio


if __name__ == "__main__":
    main()