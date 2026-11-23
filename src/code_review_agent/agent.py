"""Agent workflow. Day 2: shared state. Day 3: Planner node."""
import json
from operator import add
from pathlib import Path
from typing import Annotated, Literal, TypedDict

Status = Literal[
    "planning", "coding", "testing", "reviewing",
    "verifying", "awaiting_approval", "done", "failed",
]


class AgentState(TypedDict, total=False):
    # Input
    task: str            # e.g. "Add a priority field to todos"
    repo_path: str       # repo the agent edits (NOT this project)
    branch: str          # e.g. ai-agent/add-priority

    # Planning
    plan: dict           # {"objective", "requirements", "tasks", "files_to_change"}
    relevant_files: list[str]

    # Results
    diff: str
    test_output: str
    tests_passed: bool
    lint_passed: bool
    review_issues: list[dict]    # [{"severity", "message"}]

    # Loop control
    iteration: int
    max_iterations: int
    status: Status
    # Annotated[..., add]: new failures are appended, not overwritten
    failure_history: Annotated[list[str], add]


def initial_state(task: str, repo_path: str, max_iterations: int = 3) -> AgentState:
    """Fresh state for a new run."""
    return AgentState(
        task=task,
        repo_path=repo_path,
        branch="",
        plan={},
        relevant_files=[],
        diff="",
        test_output="",
        tests_passed=False,
        lint_passed=False,
        review_issues=[],
        iteration=0,
        max_iterations=max_iterations,
        status="planning",
        failure_history=[],
    )


# ---------------------------------------------------------------- Planner (Day 3)

PLANNER_SYSTEM = """You are a software planning assistant.
Given a coding task and the list of files in a repository, produce a short plan.
Do NOT write code. Only plan.

Return ONLY valid JSON, no explanation, no markdown, in exactly this shape:
{
  "objective": "one sentence",
  "requirements": ["what must be true when the task is done", "..."],
  "tasks": ["ordered, concrete steps", "..."],
  "files_to_change": ["paths from the file list that need edits", "..."]
}
Only list files that appear in the given file list, or new files you want to create."""

PLAN_KEYS = ("objective", "requirements", "tasks", "files_to_change")
SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}


def _get_chat():
    # Imported lazily so this file can be imported (and tested) without API keys.
    from code_review_agent.llm import chat
    return chat


def repo_tree(repo_path: str, limit: int = 50) -> list[str]:
    """Python files in the repo (relative paths), skipping venv/cache folders."""
    root = Path(repo_path)
    files = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if not (set(p.relative_to(root).parts) & SKIP_DIRS)
    ]
    return sorted(files)[:limit]


def extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a model reply (handles ``` fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model reply")
    return json.loads(text[start : end + 1])


def validate_plan(plan: dict) -> dict:
    missing = [k for k in PLAN_KEYS if k not in plan]
    if missing:
        raise ValueError(f"Plan is missing keys: {missing}")
    for k in ("requirements", "tasks", "files_to_change"):
        if not isinstance(plan[k], list):
            raise ValueError(f"'{k}' must be a list")
    return plan


async def planner_node(state: AgentState) -> dict:
    """Task (+ earlier failures) -> JSON plan. Does not touch any code."""
    chat = _get_chat()

    user = f"Task: {state['task']}\n\nFiles in the repo:\n" + "\n".join(
        repo_tree(state["repo_path"])
    )
    failures = state.get("failure_history") or []
    if failures:
        user += "\n\nPrevious attempts failed. Fix the plan using this:\n" + "\n---\n".join(
            failures[-2:]
        )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user},
    ]

    last_error = ""
    for _ in range(2):  # one retry if the model returns bad JSON
        resp = await chat(messages, model="smart", temperature=0.2)
        try:
            plan = validate_plan(extract_json(resp.choices[0].message.content))
            return {
                "plan": plan,
                "relevant_files": plan["files_to_change"],
                "status": "coding",
            }
        except (ValueError, json.JSONDecodeError) as e:
            last_error = str(e)
            messages.append({"role": "assistant", "content": resp.choices[0].message.content or ""})
            messages.append({"role": "user", "content": f"Invalid: {last_error}. Return ONLY the JSON."})

    return {"status": "failed", "failure_history": [f"Planner failed: {last_error}"]}


if __name__ == "__main__":
    # Live check (needs GROQ_API_KEY). Run from the project root:
    #   uv run python -m code_review_agent.agent
    import asyncio

    s = initial_state(
        "Add a priority field (low, medium, high) to todos",
        "eval/sample_repos/todo_api",
    )
    print(json.dumps(asyncio.run(planner_node(s)), indent=2))