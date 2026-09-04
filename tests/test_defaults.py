from __future__ import annotations

import re
import tomllib
from pathlib import Path

from or_pr_review import __version__
from or_pr_review.harness import DEFAULT_MAX_TOOL_TURNS
from or_pr_review.models import DEFAULT_JUDGE_MODEL, DEFAULT_MODEL


def _input_default(text: str, name: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == f"{name}:":
            for follow in lines[index + 1 : index + 12]:
                stripped = follow.strip()
                if stripped.startswith("default:"):
                    return stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if stripped and not stripped.startswith("#") and ":" in stripped:
                    if stripped.split(":", 1)[0] in {
                        "description",
                        "required",
                        "type",
                    }:
                        continue
                    break
    return None


def _metadata_keys(text: str, section: str) -> set[str]:
    """Top-level action metadata keys below ``inputs`` or ``outputs``."""
    lines = text.splitlines()
    start = lines.index(f"{section}:") + 1
    keys: set[str] = set()
    for line in lines[start:]:
        if line and not line.startswith((" ", "\t", "#")):
            break
        match = re.match(r"^  ([A-Za-z_][A-Za-z0-9_]*):\s*$", line)
        if match:
            keys.add(match.group(1))
    return keys


def _readme_table_keys(text: str, heading: str) -> set[str]:
    block = text.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `([^`]+)` \|", block, flags=re.MULTILINE))


def test_action_yml_default_max_tool_turns_is_fifty() -> None:
    root = Path(__file__).resolve().parents[1]
    action = (root / "action.yml").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/pr-review.yml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert _input_default(action, "max_tool_turns") == "50"
    assert _input_default(workflow, "max_tool_turns") == "50"
    assert _input_default(action, "judge_model") == DEFAULT_JUDGE_MODEL
    assert _input_default(workflow, "judge_model") == DEFAULT_JUDGE_MODEL
    assert _input_default(action, "models") == DEFAULT_MODEL
    assert _input_default(workflow, "models") == DEFAULT_MODEL
    assert _input_default(action, "github_timeout_seconds") == "120"
    assert _input_default(workflow, "github_timeout_seconds") == "120"
    assert _input_default(action, "openrouter_timeout_seconds") == "180"
    assert _input_default(workflow, "openrouter_timeout_seconds") == "180"
    assert _input_default(action, "job_budget_seconds") == "1320"
    assert _input_default(workflow, "job_budget_seconds") == "1320"
    assert _input_default(action, "all_role_deadline_seconds") == ""
    assert _input_default(workflow, "all_role_deadline_seconds") == ""
    assert "max_tool_turns: ${{ inputs.max_tool_turns }}" in workflow
    for name in (
        "github_timeout_seconds",
        "openrouter_timeout_seconds",
        "job_budget_seconds",
        "all_role_deadline_seconds",
    ):
        assert workflow.count(f"{name}: ${{{{ inputs.{name} }}}}") == 3
    # PAT/App-token callers must be able to keep the review loop working.
    assert workflow.count("bot_login: ${{ inputs.bot_login }}") == 2
    assert "| `max_tool_turns` | `50` |" in readme
    assert DEFAULT_MAX_TOOL_TURNS == 50
    assert "fail_on" in action and 'default: "never"' in action
    assert "XAI_API_KEY" not in action
    assert "api.x.ai" not in action


def test_action_metadata_contract_is_documented() -> None:
    root = Path(__file__).resolve().parents[1]
    action = (root / "action.yml").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/pr-review.yml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert _metadata_keys(action, "inputs") <= _readme_table_keys(readme, "Inputs")
    assert _metadata_keys(action, "outputs") <= _readme_table_keys(readme, "Outputs")
    assert "OPENROUTER_API_KEY" in readme
    assert "XAI_API_KEY" not in action
    assert "github.job_workflow_sha" in workflow


def test_package_version_sources_agree() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in project["project"]["dynamic"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "or_pr_review.__version__"
    }
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
