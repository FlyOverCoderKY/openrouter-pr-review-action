from __future__ import annotations

from pathlib import Path

from or_pr_review.harness import DEFAULT_MAX_TOOL_TURNS
from or_pr_review.models import DEFAULT_JUDGE_MODEL


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


def test_action_yml_default_max_tool_turns_is_fifty() -> None:
    root = Path(__file__).resolve().parents[1]
    action = (root / "action.yml").read_text(encoding="utf-8")
    workflow = (root / ".github/workflows/pr-review.yml").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert _input_default(action, "max_tool_turns") == "50"
    assert _input_default(workflow, "max_tool_turns") == "50"
    assert _input_default(action, "judge_model") == DEFAULT_JUDGE_MODEL
    assert _input_default(workflow, "judge_model") == DEFAULT_JUDGE_MODEL
    assert 'max_tool_turns: ${{ inputs.max_tool_turns }}' in workflow
    # PAT/App-token callers must be able to keep the review loop working.
    assert workflow.count("bot_login: ${{ inputs.bot_login }}") == 2
    assert "| `max_tool_turns` | `50` |" in readme
    assert DEFAULT_MAX_TOOL_TURNS == 50
    assert "fail_on" in action and 'default: "never"' in action
    assert "XAI_API_KEY" not in action
    assert "api.x.ai" not in action
