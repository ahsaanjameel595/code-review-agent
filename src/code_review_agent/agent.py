"""Agent workflow. Day 2: state. Day 3: Planner. Day 5: Coder."""
import difflib
import json
import sys
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
        plan={},
        branch="", 
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


# ----------------------------------------------------------------- Coder (Day 5)

CODER_SYSTEM = """You are a careful software engineer. You edit a Python repository using tools.
Rules:
1. Start with list_files, then read_file every file you will change. Use search_code to find things.
2. Make the smallest change that satisfies the plan. Do not rewrite unrelated code.
3. write_file replaces the WHOLE file, so always send the complete new file content.
4. If you add or change behavior, add or update tests in the repo's test file.
5. Do not touch files outside the plan unless it is really necessary.
6. You cannot run tests. Another step does that.
When you are done, reply with 2-3 lines summarizing what you changed."""

# Tried in order. If one hits a rate limit or fails a tool call, the next one runs.
CODER_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini"}


def build_coder_messages(state: AgentState) -> list[tuple[str, str]]:
    user = f"Task: {state['task']}\n\nPlan:\n{json.dumps(state.get('plan', {}), indent=2)}"
    if state.get("relevant_files"):
        user += "\n\nFiles to change: " + ", ".join(state["relevant_files"])
    failures = state.get("failure_history") or []
    if failures:
        user += "\n\nEarlier attempts failed:\n" + "\n---\n".join(failures[-2:])
    if state.get("test_output") and not state.get("tests_passed"):
        user += "\n\nLatest failing test output:\n" + state["test_output"][-3000:]
    return [("system", CODER_SYSTEM), ("user", user)]


def snapshot_repo(repo_path: str, max_bytes: int = 200_000) -> dict[str, str]:
    """Map of relative path -> text for editable files (used to build the diff)."""
    from code_review_agent.mcp_server import _iter_files  # same skip rules as the tools

    root = Path(repo_path).resolve()
    snap: dict[str, str] = {}
    for p in _iter_files(root):
        if p.suffix.lower() in TEXT_SUFFIXES and p.stat().st_size <= max_bytes:
            snap[p.relative_to(root).as_posix()] = p.read_text(encoding="utf-8", errors="replace")
    return snap


def make_diff(before: dict[str, str], after: dict[str, str], limit: int = 20_000) -> str:
    """Unified diff between two snapshots (new, changed and deleted files)."""
    chunks = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        if old == new:
            continue
        chunks.extend(
            difflib.unified_diff(
                (old or "").splitlines(keepends=True),
                (new or "").splitlines(keepends=True),
                fromfile=f"a/{name}" if old is not None else "/dev/null",
                tofile=f"b/{name}" if new is not None else "/dev/null",
            )
        )
        if chunks and not chunks[-1].endswith("\n"):
            chunks.append("\n")
    return "".join(chunks)[:limit]


def _is_retryable(exc: Exception) -> bool:
    """Rate limits and broken tool calls -> try the next model."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        s in text
        for s in ("ratelimit", "rate limit", "429", "tool_use_failed",
                  "failed to call a function", "tool call validation")
    )


async def run_with_fallback(run_one, models: list[str]):
    """Run run_one(model) for each model until one works. Returns (model, result)."""
    last: Exception | None = None
    for model in models:
        try:
            return model, await run_one(model)
        except Exception as e:  # noqa: BLE001
            if not _is_retryable(e):
                raise
            last = e
    raise RuntimeError(f"All coder models failed. Last error: {last}")


def _agent_factory():
    try:
        from langchain.agents import create_agent  # newer LangChain / LangGraph
        return create_agent
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent


async def _run_agent(model_name: str, repo_path: str, messages) -> str:
    """One Coder run: MCP file tools + a Groq model in a tool-calling loop."""
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq
    from langchain_mcp_adapters.client import MultiServerMCPClient

    load_dotenv()
    client = MultiServerMCPClient(
        {
            "repo_tools": {
                "command": sys.executable,  # the venv's python
                "args": ["-m", "code_review_agent.mcp_server"],
                "transport": "stdio",
                "env": {"TARGET_REPO": str(Path(repo_path).resolve())},
            }
        }
    )
    tools = await client.get_tools()
    agent = _agent_factory()(ChatGroq(model=model_name, temperature=0.2), tools)
    result = await agent.ainvoke({"messages": messages}, config={"recursion_limit": 40})
    return str(result["messages"][-1].content)


async def coder_node(state: AgentState) -> dict:
    """Edit the repo following the plan. Returns the diff and moves to testing."""
    repo = state["repo_path"]
    iteration = state.get("iteration", 0) + 1
    before = snapshot_repo(repo)
    messages = build_coder_messages(state)

    try:
        await run_with_fallback(lambda m: _run_agent(m, repo, messages), CODER_MODELS)
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "iteration": iteration,
                "failure_history": [f"Coder failed: {e}"]}

    diff = make_diff(before, snapshot_repo(repo))
    if not diff:
        return {"status": "failed", "iteration": iteration,
                "failure_history": ["Coder made no changes"]}
    return {"diff": diff, "status": "testing", "iteration": iteration}


if __name__ == "__main__":
    # Live check (needs GROQ_API_KEY). Run from the project root:
    #   uv run python -m code_review_agent.agent
    # It EDITS eval/sample_repos/todo_api. Undo with:
    #   git checkout -- eval/sample_repos/todo_api
    import asyncio

    async def demo():
        s = initial_state(
            "Add a priority field (low, medium, high; default medium) to todos",
            "eval/sample_repos/todo_api",
        )
        s.update(await planner_node(s))
        print("PLAN:", json.dumps(s.get("plan", {}), indent=2))
        if s["status"] == "failed":
            print("Planner failed:", s["failure_history"]); return
        s.update(await coder_node(s))
        print("STATUS:", s["status"], "| iteration:", s["iteration"])
        print(s.get("diff") or s.get("failure_history"))

    asyncio.run(demo())