import asyncio
from types import SimpleNamespace

import pytest

from code_review_agent import agent
from code_review_agent.agent import extract_json, initial_state, planner_node, validate_plan

GOOD_PLAN = (
    '{"objective": "Add priority", "requirements": ["todos have priority"], '
    '"tasks": ["edit model", "edit endpoints"], "files_to_change": ["main.py"]}'
)


def fake_reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def make_repo(tmp_path):
    (tmp_path / "main.py").write_text("x = 1")
    return str(tmp_path)


def test_extract_json_handles_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_raises_without_json():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_validate_plan_rejects_missing_keys():
    with pytest.raises(ValueError):
        validate_plan({"objective": "x"})


def test_planner_returns_plan(monkeypatch, tmp_path):
    async def fake_chat(messages, **kw):
        return fake_reply("```json\n" + GOOD_PLAN + "\n```")

    monkeypatch.setattr(agent, "_get_chat", lambda: fake_chat)
    out = asyncio.run(planner_node(initial_state("add priority", make_repo(tmp_path))))
    assert out["status"] == "coding"
    assert out["relevant_files"] == ["main.py"]


def test_planner_retries_then_succeeds(monkeypatch, tmp_path):
    replies = iter(["not json", GOOD_PLAN])

    async def fake_chat(messages, **kw):
        return fake_reply(next(replies))

    monkeypatch.setattr(agent, "_get_chat", lambda: fake_chat)
    out = asyncio.run(planner_node(initial_state("add priority", make_repo(tmp_path))))
    assert out["status"] == "coding"


def test_planner_fails_after_two_bad_replies(monkeypatch, tmp_path):
    async def fake_chat(messages, **kw):
        return fake_reply("still not json")

    monkeypatch.setattr(agent, "_get_chat", lambda: fake_chat)
    out = asyncio.run(planner_node(initial_state("add priority", make_repo(tmp_path))))
    assert out["status"] == "failed"
    assert "Planner failed" in out["failure_history"][0]