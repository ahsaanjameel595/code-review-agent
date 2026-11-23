from code_review_agent.agent import initial_state


def test_initial_state_defaults():
    state = initial_state("Add a feature", "C:/repo")

    assert state["task"] == "Add a feature"
    assert state["repo_path"] == "C:/repo"
    assert state["branch"] == ""
    assert state["plan"] == {}
    assert state["relevant_files"] == []
    assert state["diff"] == ""
    assert state["test_output"] == ""
    assert state["tests_passed"] is False
    assert state["lint_passed"] is False
    assert state["review_issues"] == []
    assert state["iteration"] == 0
    assert state["max_iterations"] == 3
    assert state["status"] == "planning"
    assert state["failure_history"] == []


def test_initial_state_accepts_custom_max_iterations():
    state = initial_state("Fix a bug", "C:/repo", max_iterations=5)

    assert state["max_iterations"] == 5


def test_initial_state_returns_independent_mutable_values():
    first = initial_state("Task one", "C:/repo")
    second = initial_state("Task two", "C:/repo")
    first["review_issues"].append({"severity": "low", "message": "test"})
    first["failure_history"].append("failure")

    assert second["review_issues"] == []
    assert second["failure_history"] == []
