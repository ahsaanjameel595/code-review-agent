import asyncio
from pathlib import Path

import pytest

from code_review_agent import agent
from code_review_agent.agent import (
    build_coder_messages, coder_node, initial_state, make_diff,
    run_with_fallback, snapshot_repo,
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "main.py").write_text("x = 1\n")
    (tmp_path / ".env").write_text("SECRET=1")
    return tmp_path


def test_snapshot_skips_secrets(repo):
    assert set(snapshot_repo(str(repo))) == {"main.py"}


def test_make_diff_changed_new_deleted():
    d = make_diff({"a.py": "x = 1\n", "gone.py": "y\n"}, {"a.py": "x = 2\n", "new.py": "z\n"})
    assert "-x = 1" in d and "+x = 2" in d
    assert "+++ b/new.py" in d and "--- a/gone.py" in d


def test_make_diff_empty_when_equal():
    assert make_diff({"a.py": "x\n"}, {"a.py": "x\n"}) == ""


def test_messages_include_failures_and_test_output(repo):
    s = initial_state("t", str(repo))
    s["failure_history"] = ["test_x failed"]
    s["test_output"] = "AssertionError: boom"
    user = build_coder_messages(s)[1][1]
    assert "test_x failed" in user and "AssertionError: boom" in user


def test_coder_node_success(repo, monkeypatch):
    async def fake(model, repo_path, messages):
        (Path(repo_path) / "main.py").write_text("x = 2\n")
        return "done"

    monkeypatch.setattr(agent, "_run_agent", fake)
    out = asyncio.run(coder_node(initial_state("t", str(repo))))
    assert out["status"] == "testing" and out["iteration"] == 1
    assert "-x = 1" in out["diff"] and "+x = 2" in out["diff"]


def test_coder_node_no_changes_is_failure(repo, monkeypatch):
    async def fake(model, repo_path, messages):
        return "nothing"

    monkeypatch.setattr(agent, "_run_agent", fake)
    out = asyncio.run(coder_node(initial_state("t", str(repo))))
    assert out["status"] == "failed" and "no changes" in out["failure_history"][0]


def test_fallback_moves_to_next_model_on_rate_limit():
    class RateLimitError(Exception):
        pass

    async def run_one(model):
        if model == "a":
            raise RateLimitError("429 too many requests")
        return f"ok from {model}"

    assert asyncio.run(run_with_fallback(run_one, ["a", "b"])) == ("b", "ok from b")


def test_fallback_raises_other_errors_immediately():
    async def run_one(model):
        raise ValueError("bug in my code")

    with pytest.raises(ValueError):
        asyncio.run(run_with_fallback(run_one, ["a", "b"]))


def test_all_models_rate_limited(repo, monkeypatch):
    async def fake(model, repo_path, messages):
        raise Exception("429 rate limit")

    monkeypatch.setattr(agent, "_run_agent", fake)
    out = asyncio.run(coder_node(initial_state("t", str(repo))))
    assert out["status"] == "failed" and "All coder models failed" in out["failure_history"][0]