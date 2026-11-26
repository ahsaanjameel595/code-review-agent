import asyncio

import pytest

from code_review_agent import mcp_server as s


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text("def hello():\n    return 'hi'\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "util.py").write_text("HELLO = 1\n")
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "skip.py").write_text("hello = 'skip me'")
    monkeypatch.setenv("TARGET_REPO", str(tmp_path))
    return tmp_path


def test_list_files_skips_blocked(repo):
    files = s.list_files()["files"]
    assert "main.py" in files and "pkg/util.py" in files
    assert ".env" not in files and ".venv/skip.py" not in files


def test_read_file(repo):
    out = s.read_file("main.py")
    assert "def hello" in out["content"] and out["truncated"] is False


def test_read_outside_repo_blocked(repo):
    with pytest.raises(ValueError):
        s.read_file("../outside.txt")


def test_read_env_blocked(repo):
    with pytest.raises(ValueError):
        s.read_file(".env")


def test_write_creates_and_overwrites(repo):
    r1 = s.write_file("pkg/new.py", "x = 1\n")
    assert r1["created"] is True
    assert (repo / "pkg" / "new.py").read_text() == "x = 1\n"
    r2 = s.write_file("pkg/new.py", "x = 2\n")
    assert r2["created"] is False


def test_write_bad_suffix_blocked(repo):
    with pytest.raises(ValueError):
        s.write_file("run.exe", "nope")


def test_write_outside_repo_blocked(repo):
    with pytest.raises(ValueError):
        s.write_file("../evil.py", "x")


def test_search_finds_and_skips_venv(repo):
    files = {m["file"] for m in s.search_code("hello")["matches"]}
    assert files == {"main.py", "pkg/util.py"}


def test_search_regex_and_limit(repo):
    out = s.search_code(r"def \w+", regex=True)
    assert out["matches"][0]["line"] == 1
    assert s.search_code("hello", max_results=1)["truncated"] is True


def test_server_registers_tools(repo):
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    async def run():
        async with Client(s.build_server()) as c:
            return {t.name for t in await c.list_tools()}

    assert asyncio.run(run()) >= {"list_files", "read_file", "write_file", "search_code"}