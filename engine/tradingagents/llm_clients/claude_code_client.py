"""Claude through the local Claude Code CLI (``claude -p``), on the person's Claude plan.

Each chat call runs one headless ``claude -p`` turn with every built-in tool,
MCP server and settings file switched off and the agent's own system prompt in
place of Claude Code's, so the model sees the same prompt the Anthropic
provider would send. Tool calls and structured output come back through
``--json-schema``; LangGraph still executes the tools, exactly as upstream.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel

from .base_client import BaseLLMClient
from .validators import validate_model

# The plan's usage cap (a 429 like a passing rate limit, but it lasts hours).
_PLAN_LIMIT = re.compile(r"usage limit|hit your limit", re.IGNORECASE)

_ROLES = {"human": "User", "ai": "Assistant", "tool": "Tool result", "system": "System"}

_TOOL_INSTRUCTIONS = """

## Tool use
The tools listed below are live and available to you. You call them by listing them in the "tool_calls" field of your structured reply: each entry names a tool and gives its arguments. The system runs every call and returns the results to you in the next turn. Do not claim a tool is unavailable; to get data, call the tool. While calling tools, leave "content" empty. Once you have what you need, write your answer in "content" and leave "tool_calls" empty. Put the whole answer in "content": anything written outside the structured reply is discarded, so never point to text "above".

Tools:
{tools}"""

_RESEND = ('\n\n[System] Your last reply put only a short note in "content", and anything written outside '
           '"content" is discarded. Reply again with your complete final answer in "content".')


def _lost_answer(out: dict) -> bool:
    """A final reply with a stub for content: the model wrote its report outside the structured reply."""
    s = out.get("structured_output")
    return isinstance(s, dict) and not s.get("tool_calls") and len(str(s.get("content") or "").strip()) < 300


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _transcript(messages: list[BaseMessage]) -> tuple[str, str]:
    """The system prompt and the rest of the conversation written out as one prompt."""
    system, turns = [], []
    for m in messages:
        if isinstance(m, SystemMessage):
            system.append(_text(m.content))
            continue
        body = _text(m.content)
        if isinstance(m, AIMessage) and m.tool_calls:
            calls = "; ".join(f"{c['name']}({json.dumps(c['args'], default=str)})" for c in m.tool_calls)
            body = f"{body}\n[Called {calls}]".strip()
        if isinstance(m, ToolMessage):
            body = f"[{m.name or 'tool'}]\n{body}"
        turns.append(f"{_ROLES.get(m.type, m.type)}:\n{body}")
    if len(turns) == 1 and messages[-1].type == "human":
        return "\n\n".join(system), _text(messages[-1].content)
    return "\n\n".join(system), "\n\n".join(turns)


def _tool_schema(names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": names}, "args": {"type": "object"}},
                    "required": ["name", "args"],
                },
            },
        },
        "required": ["content", "tool_calls"],
    }


class ChatClaudeCode(BaseChatModel):
    """A LangChain chat model whose every call is one ``claude -p`` turn."""

    model: str
    effort: str | None = None
    max_retries: int = 2
    timeout: float | None = 900
    cli_path: str = "claude"

    @property
    def _llm_type(self) -> str:
        return "claude_code"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(t)["function"] for t in tools], **kwargs)

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        if include_raw:
            raise NotImplementedError("include_raw is not supported by the claude_code provider")
        is_model = isinstance(schema, type) and issubclass(schema, BaseModel)
        json_schema = schema.model_json_schema() if is_model else convert_to_openai_tool(schema)["function"]["parameters"]

        def parse(message: AIMessage):
            data = message.additional_kwargs.get("structured_output")
            return schema.model_validate(data) if is_model and data is not None else data

        return self.bind(response_schema=json_schema) | RunnableLambda(parse)

    def _run(self, system: str, prompt: str, schema: dict | None) -> dict:
        cmd = [
            self.cli_path, "-p", "--output-format", "json", "--model", self.model,
            "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--no-session-persistence", "--disable-slash-commands",
            # Claude Code's own system prompt would change what the agents are told.
            "--system-prompt", system or "You are a helpful assistant.",
        ]
        if self.effort:
            cmd += ["--effort", self.effort]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
        # A neutral cwd so no project CLAUDE.md is read into the prompt.
        cwd = tempfile.gettempdir()
        # A set ANTHROPIC_API_KEY (e.g. from .env) would bill the API instead of the plan.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        error = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(5 * 2 ** (attempt - 1))
            try:
                proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                      timeout=self.timeout, cwd=cwd, env=env)
            except subprocess.TimeoutExpired as exc:
                # A hung call rarely recovers; retrying would stall the run another timeout each time.
                raise RuntimeError(f"claude -p timed out after {exc.timeout}s") from exc
            try:
                out = json.loads(proc.stdout)
            except json.JSONDecodeError:
                error = f"exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:500]}"
                continue
            if not out.get("is_error"):
                return out
            error = f"{out.get('subtype')}: {str(out.get('result'))[:500]}"
            status = out.get("api_error_status") or 0
            # Auth, bad-model and plan-limit errors won't clear on a retry; fail fast.
            if (400 <= status < 500 and status != 429) or _PLAN_LIMIT.search(str(out.get("result"))):
                break
        raise RuntimeError(f"claude -p failed: {error}")

    def _generate(self, messages, stop=None, run_manager=None, tools=None, response_schema=None, **kwargs):
        system, prompt = _transcript(messages)
        schema = response_schema
        if tools and not schema:
            system += _TOOL_INSTRUCTIONS.format(tools=json.dumps(tools, indent=1))
            schema = _tool_schema([t["name"] for t in tools])
        out = self._run(system, prompt, schema)
        if tools and not response_schema and _lost_answer(out):
            # claude -p keeps only the structured reply, so a report written as prose comes back as "see above".
            out = self._run(system, prompt + _RESEND, schema)

        usage = out.get("usage") or {}
        tokens_in = sum(usage.get(k, 0) for k in
                        ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        tokens_out = usage.get("output_tokens", 0)
        structured = out.get("structured_output")
        content, tool_calls, extra = out.get("result") or "", [], {}
        if response_schema:
            extra["structured_output"] = structured
        elif tools and isinstance(structured, dict):
            content = structured.get("content", "")
            tool_calls = [{"name": c["name"], "args": c.get("args") or {}, "id": f"call_{uuid.uuid4().hex[:24]}"}
                          for c in structured.get("tool_calls") or []]
        message = AIMessage(
            content=content, tool_calls=tool_calls, additional_kwargs=extra,
            usage_metadata={"input_tokens": tokens_in, "output_tokens": tokens_out,
                            "total_tokens": tokens_in + tokens_out},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class ClaudeCodeClient(BaseLLMClient):
    """Client for Claude models served through the local Claude Code CLI."""

    provider = "claude_code"

    def get_llm(self) -> Any:
        cli = shutil.which("claude")
        if not cli:
            raise RuntimeError("The claude_code provider needs the Claude Code CLI on PATH: "
                               "https://docs.claude.com/en/docs/claude-code")
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model, "cli_path": cli}
        # temperature and max_tokens are not settable through Claude Code; they are dropped.
        for key in ("effort", "max_retries", "timeout", "callbacks"):
            if self.kwargs.get(key) is not None:
                llm_kwargs[key] = self.kwargs[key]
        return ChatClaudeCode(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("anthropic", self.model)
