#!/usr/bin/env python3
"""Drive benchmark scenarios through a tool-calling LLM against the MCP server.

Emits trace JSONL in the exact shape benchmarks/score_trace_corpus.py consumes:

    {"scenario": str, "tools": [str, ...], "success": bool,
     "first_useful_ms": float | null, "total_ms": float}

Modes:
  --stub                 offline: a scripted policy replays each scenario's
                         preferred tool sequence against a fake executor.
                         No server, no kernel, no network, no API key.
  --provider anthropic   live: needs ANTHROPIC_API_KEY (plain HTTP via urllib).
  --provider openai      live: needs OPENAI_API_KEY, honors OPENAI_BASE_URL.
                         Live modes spawn the MCP server over stdio.

Examples:
  uv run python benchmarks/llm_driver.py --stub --profile lean -o stub_lean.jsonl
  uv run python benchmarks/llm_driver.py --provider anthropic --model claude-opus-4-8 \
      --profile lean -o live_lean.jsonl
  uv run python benchmarks/score_trace_corpus.py stub_lean.jsonl live_lean.jsonl --profile lean
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = REPO_ROOT / "benchmarks" / "scenarios.json"
SYSTEM_PROMPT = (
    "You are operating a Mathematica MCP server. Solve the user's task using the "
    "available tools, in as few tool calls as possible, then answer in plain text."
)
MAX_TURNS = 8


def load_scenarios(path: Path = SCENARIOS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def preferred_tools(scenario: dict[str, Any], profile: str) -> list[str]:
    key = "lean_preferred_tools" if profile == "lean" else "preferred_tools"
    return scenario.get(key) or scenario.get("preferred_tools", [])


def _useful(result_text: str) -> bool:
    return '"success": false' not in result_text.lower()


# ---------------------------------------------------------------- stub mode


class StubExecutor:
    """Fake tool executor: returns a canned success payload, touches nothing."""

    def call(self, name: str, args: dict[str, Any]) -> str:
        return json.dumps({"success": True, "tool": name, "stub": True})


def run_stub_scenario(scenario: dict[str, Any], profile: str, executor: Any) -> dict[str, Any]:
    """Scripted policy: replay the scenario's preferred tool sequence."""
    start = time.monotonic()
    tools: list[str] = []
    first_useful_ms: float | None = None
    success = True
    for name in preferred_tools(scenario, profile):
        try:
            result = executor.call(name, {})
        except Exception:
            success = False
            break
        tools.append(name)
        if first_useful_ms is None and _useful(str(result)):
            first_useful_ms = (time.monotonic() - start) * 1000
    return {
        "scenario": scenario["id"],
        "tools": tools,
        "success": success and bool(tools),
        "first_useful_ms": round(first_useful_ms, 3) if first_useful_ms is not None else None,
        "total_ms": round((time.monotonic() - start) * 1000, 3),
    }


def run_stub(scenarios: list[dict[str, Any]], profile: str, executor: Any | None = None) -> list[dict[str, Any]]:
    executor = executor or StubExecutor()
    return [run_stub_scenario(scenario, profile, executor) for scenario in scenarios]


# ---------------------------------------------------------------- live mode


def _http_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def _anthropic_turn(model: str, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
    return _http_json(
        "https://api.anthropic.com/v1/messages",
        {"model": model, "max_tokens": 2048, "system": SYSTEM_PROMPT, "messages": messages, "tools": tools},
        {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
    )


def _openai_turn(model: str, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return _http_json(
        f"{base}/chat/completions",
        {"model": model, "messages": messages, "tools": tools},
        {"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    )


async def _run_live_scenario(
    session: Any, scenario: dict[str, Any], provider: str, model: str, mcp_tools: list
) -> dict[str, Any]:
    start = time.monotonic()
    tools_used: list[str] = []
    first_useful_ms: float | None = None
    success = False

    async def call_tool(name: str, args: dict[str, Any]) -> str:
        nonlocal first_useful_ms
        result = await session.call_tool(name, args)
        tools_used.append(name)
        text = "".join(getattr(block, "text", "") for block in result.content)
        if first_useful_ms is None and not result.isError and _useful(text):
            first_useful_ms = (time.monotonic() - start) * 1000
        return text or "(no text output)"

    if provider == "anthropic":
        tools = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema} for t in mcp_tools]
        messages: list[dict] = [{"role": "user", "content": scenario["prompt"]}]
        for _ in range(MAX_TURNS):
            resp = _anthropic_turn(model, messages, tools)
            messages.append({"role": "assistant", "content": resp["content"]})
            if resp.get("stop_reason") != "tool_use":
                success = bool(tools_used)
                break
            results = []
            for block in resp["content"]:
                if block["type"] == "tool_use":
                    text = await call_tool(block["name"], block["input"])
                    results.append({"type": "tool_result", "tool_use_id": block["id"], "content": text[:8000]})
            messages.append({"role": "user", "content": results})
    else:
        tools = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema},
            }
            for t in mcp_tools
        ]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": scenario["prompt"]}]
        for _ in range(MAX_TURNS):
            msg = _openai_turn(model, messages, tools)["choices"][0]["message"]
            messages.append(msg)
            if not msg.get("tool_calls"):
                success = bool(tools_used)
                break
            for tc in msg["tool_calls"]:
                text = await call_tool(tc["function"]["name"], json.loads(tc["function"]["arguments"] or "{}"))
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": text[:8000]})

    return {
        "scenario": scenario["id"],
        "tools": tools_used,
        "success": success,
        "first_useful_ms": round(first_useful_ms, 3) if first_useful_ms is not None else None,
        "total_ms": round((time.monotonic() - start) * 1000, 3),
    }


async def run_live(scenarios: list[dict[str, Any]], profile: str, provider: str, model: str) -> list[dict[str, Any]]:
    key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key):
        raise SystemExit(f"live mode with --provider {provider} requires {key}")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mathematica_mcp"],
        env={**os.environ, "MATHEMATICA_PROFILE": profile},
    )
    rows = []
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        mcp_tools = (await session.list_tools()).tools
        for scenario in scenarios:
            rows.append(await _run_live_scenario(session, scenario, provider, model, mcp_tools))
    return rows


# --------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stub", action="store_true", help="offline scripted policy; no server/kernel/network")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--profile", choices=["classic", "lean"], default="lean")
    parser.add_argument("--only", help="run a single scenario id")
    parser.add_argument("-o", "--out", default="-", help="output JSONL path (default stdout)")
    args = parser.parse_args()

    scenarios = load_scenarios()
    if args.only:
        scenarios = [s for s in scenarios if s["id"] == args.only]
        if not scenarios:
            raise SystemExit(f"unknown scenario id: {args.only}")

    if args.stub:
        rows = run_stub(scenarios, args.profile)
    else:
        rows = asyncio.run(run_live(scenarios, args.profile, args.provider, args.model))

    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
