import json
import subprocess

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from tradingagents.agents.schemas import TraderProposal
from tradingagents.llm_clients.claude_code_client import ChatClaudeCode
from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client


@tool
def get_price(ticker: str) -> str:
    """Latest price for a ticker."""
    return "1"


def _fake_run(calls, structured=None, result=""):
    def run(cmd, input, **kwargs):
        calls.append((cmd, input))
        out = {"type": "result", "is_error": False, "result": result, "structured_output": structured,
               "usage": {"input_tokens": 3, "cache_read_input_tokens": 4, "output_tokens": 5}}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(out), "")
    return run


def test_plain_call_sends_the_agent_system_prompt_and_no_tools(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls, result="hello"))
    msg = ChatClaudeCode(model="claude-sonnet-5", effort="high").invoke(
        [SystemMessage("be a trader"), HumanMessage("hi")])
    cmd, prompt = calls[0]
    assert msg.content == "hello" and prompt == "hi"
    assert cmd[cmd.index("--system-prompt") + 1] == "be a trader"
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--effort") + 1] == "high"
    assert msg.usage_metadata["input_tokens"] == 7


def test_an_api_key_in_the_environment_is_not_passed_to_claude(monkeypatch):
    seen = {}

    def run(cmd, input, env, **kwargs):
        seen.update(env)
        return _fake_run([])(cmd, input)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(subprocess, "run", run)
    ChatClaudeCode(model="claude-sonnet-5").invoke("hi")
    assert "ANTHROPIC_API_KEY" not in seen and "PATH" in seen


def test_bound_tools_come_back_as_tool_calls_and_history_is_written_out(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(
        calls, structured={"content": "", "tool_calls": [{"name": "get_price", "args": {"ticker": "NVDA"}}]}))
    history = [HumanMessage("price?"),
               AIMessage("", tool_calls=[{"name": "get_price", "args": {"ticker": "AMD"}, "id": "c1"}]),
               ToolMessage("2", tool_call_id="c1", name="get_price")]
    msg = ChatClaudeCode(model="claude-sonnet-5").bind_tools([get_price]).invoke(history)
    cmd, prompt = calls[0]
    assert msg.tool_calls[0]["name"] == "get_price" and msg.tool_calls[0]["args"] == {"ticker": "NVDA"}
    assert '"enum": ["get_price"]' in cmd[cmd.index("--json-schema") + 1]
    assert 'get_price({"ticker": "AMD"})' in prompt and "[get_price]\n2" in prompt


def test_a_final_answer_left_outside_content_is_asked_for_again(monkeypatch):
    calls, replies = [], [{"content": "Report compiled above.", "tool_calls": []},
                          {"content": "# Fundamentals\n" + "x" * 400, "tool_calls": []}]

    def run(cmd, input, **kwargs):
        return _fake_run(calls, structured=replies[len(calls)])(cmd, input)

    monkeypatch.setattr(subprocess, "run", run)
    msg = ChatClaudeCode(model="claude-sonnet-5").bind_tools([get_price]).invoke("report?")
    assert len(calls) == 2 and "[System] Your last reply" in calls[1][1]
    assert msg.content.startswith("# Fundamentals") and not msg.tool_calls


def test_structured_output_parses_into_the_schema(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls, structured={"action": "Buy", "reasoning": "r"}))
    out = ChatClaudeCode(model="claude-sonnet-5").with_structured_output(TraderProposal).invoke("go")
    assert isinstance(out, TraderProposal) and out.action == "Buy"


def _error_run(calls, result, status=None):
    def run(cmd, input, **kwargs):
        calls.append(cmd)
        out = {"type": "result", "is_error": True, "subtype": "error", "result": result,
               "api_error_status": status}
        return subprocess.CompletedProcess(cmd, 1, json.dumps(out), "")
    return run


@pytest.mark.parametrize("result,status", [("Claude AI usage limit reached|1700000000", 429),
                                           ("You've hit your limit · resets 5pm", None),
                                           ("Invalid API key · Please run /login", 401)])
def test_errors_a_retry_cannot_fix_raise_at_once(monkeypatch, result, status):
    calls, sleeps = [], []
    monkeypatch.setattr(subprocess, "run", _error_run(calls, result, status))
    monkeypatch.setattr("time.sleep", sleeps.append)
    with pytest.raises(RuntimeError, match=result.split(" ")[0]):
        ChatClaudeCode(model="claude-sonnet-5", max_retries=2).invoke("go")
    assert len(calls) == 1 and sleeps == []


def test_a_transient_error_is_retried_with_backoff_then_raises(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(subprocess, "run", _error_run(calls, "Overloaded", 529))
    monkeypatch.setattr("time.sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="Overloaded"):
        ChatClaudeCode(model="claude-sonnet-5", max_retries=2).invoke("go")
    assert len(calls) == 3 and sleeps == [5, 10]


def test_a_timeout_raises_without_retrying(monkeypatch):
    def run(cmd, input, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, 900)
    calls = []
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="timed out"):
        ChatClaudeCode(model="claude-sonnet-5", max_retries=2).invoke("go")
    assert len(calls) == 1


def test_factory_routes_claude_code_with_effort(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/claude")
    kwargs = build_llm_kwargs({"llm_provider": "claude_code", "anthropic_effort": "low", "temperature": 0.1})
    llm = create_llm_client("claude_code", "claude-sonnet-5", **kwargs).get_llm()
    assert isinstance(llm, ChatClaudeCode) and llm.effort == "low"
