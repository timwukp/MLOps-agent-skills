#!/usr/bin/env python3
"""Tool-using LLM agent builder with ReAct-style reasoning loop.

Usage:
    python build_agent.py --system-prompt "You are a helpful assistant." --tools all
    python build_agent.py --tools calculator,python --max-iterations 5
    python build_agent.py --model gpt-4o --tools all --human-in-loop
    python build_agent.py --system-prompt "Research assistant" --tools file,calculator
"""
import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """Schema for a callable tool the agent can invoke."""
    name: str
    description: str
    parameters: Dict[str, Any]
    callable_fn: Callable[..., str]


def tool_web_search(query: str, num_results: int = 3) -> str:
    """Mock web search - returns placeholder results."""
    logger.info(f"[web_search] query={query!r} num_results={num_results}")
    results = [
        {"title": f"Result {i+1} for '{query}'", "snippet": f"Simulated snippet {i+1} about {query}."}
        for i in range(num_results)
    ]
    return json.dumps(results, indent=2)


def tool_calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    logger.info(f"[calculator] expression={expression!r}")
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("_")
    }
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def tool_python_executor(code: str) -> str:
    """Execute a restricted Python snippet and capture its printed output."""
    logger.info(f"[python_executor] code length={len(code)}")
    blocked = ["import os", "import sys", "subprocess", "__import__", "open(", "exec(", "eval("]
    for pattern in blocked:
        if pattern in code:
            return f"Blocked: '{pattern}' is not allowed in restricted mode."
    from io import StringIO
    buf = StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len,  # noqa: S102
                                      "int": int, "float": float, "str": str,
                                      "list": list, "dict": dict, "sorted": sorted,
                                      "enumerate": enumerate, "zip": zip, "sum": sum,
                                      "min": min, "max": max, "abs": abs, "round": round,
                                      "math": math, "True": True, "False": False, "None": None}})
        return buf.getvalue() or "(no output)"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        sys.stdout = old_stdout


def tool_file_reader(path: str, max_lines: int = 50) -> str:
    """Read the first N lines of a local file."""
    logger.info(f"[file_reader] path={path!r} max_lines={max_lines}")
    target = Path(path)
    if not target.exists():
        return f"Error: file not found: {path}"
    try:
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()[:max_lines]
        return "\n".join(lines)
    except Exception as exc:
        return f"Error reading file: {exc}"


BUILTIN_TOOLS: Dict[str, ToolDefinition] = {
    "web_search": ToolDefinition(
        name="web_search",
        description="Search the web for information. Returns a JSON list of results.",
        parameters={"query": {"type": "string", "required": True},
                    "num_results": {"type": "integer", "default": 3}},
        callable_fn=tool_web_search,
    ),
    "calculator": ToolDefinition(
        name="calculator",
        description="Evaluate a mathematical expression (e.g. '2**10 + math.sqrt(144)').",
        parameters={"expression": {"type": "string", "required": True}},
        callable_fn=tool_calculator,
    ),
    "python": ToolDefinition(
        name="python_executor",
        description="Execute a short Python snippet in a restricted sandbox and return printed output.",
        parameters={"code": {"type": "string", "required": True}},
        callable_fn=tool_python_executor,
    ),
    "file": ToolDefinition(
        name="file_reader",
        description="Read the first N lines of a local file.",
        parameters={"path": {"type": "string", "required": True},
                    "max_lines": {"type": "integer", "default": 50}},
        callable_fn=tool_file_reader,
    ),
}


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

class Memory:
    """Sliding-window conversation memory with summarization trigger."""

    def __init__(self, window_size: int = 20, summarize_threshold: int = 30):
        self.messages: List[Dict[str, str]] = []
        self.window_size = window_size
        self.summarize_threshold = summarize_threshold
        self.summaries: List[str] = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) >= self.summarize_threshold:
            self._summarize_old()

    def _summarize_old(self):
        """Compress old messages into a summary string."""
        overflow = self.messages[: -self.window_size]
        summary_parts = []
        for msg in overflow:
            tag = msg["role"].upper()
            snippet = msg["content"][:200]
            summary_parts.append(f"[{tag}] {snippet}")
        summary = "Previous conversation summary:\n" + "\n".join(summary_parts)
        self.summaries.append(summary)
        self.messages = self.messages[-self.window_size:]
        logger.info(f"Memory compacted: kept {len(self.messages)} messages, {len(self.summaries)} summaries")

    def get_context(self) -> List[Dict[str, str]]:
        """Return messages suitable for the LLM, prepending summaries if any."""
        context = []
        if self.summaries:
            combined = "\n---\n".join(self.summaries)
            context.append({"role": "system", "content": combined})
        context.extend(self.messages)
        return context


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Structured output from an agent run."""
    answer: str
    reasoning_trace: List[str]
    tools_used: List[str]
    iterations: int


class Agent:
    """ReAct-style tool-using LLM agent."""

    REACT_INSTRUCTION = (
        "You are a ReAct agent. For each step, output EXACTLY one of the following formats:\n"
        "Thought: <your reasoning>\n"
        "Action: <tool_name>(<json_args>)\n"
        "Answer: <your final answer>\n\n"
        "Available tools:\n{tool_descriptions}\n\n"
        "Always start with a Thought, then decide on an Action or give the final Answer."
    )

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[ToolDefinition]] = None,
        model: str = "gpt-4o-mini",
        max_iterations: int = 10,
        human_in_loop: bool = False,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in (tools or [])}
        self.model = model
        self.max_iterations = max_iterations
        self.human_in_loop = human_in_loop
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.memory = Memory()

    # -- LLM call (OpenAI-compatible) ---------------------------------------
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("Install: pip install openai")
            sys.exit(1)
        client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2, max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    # -- Tool descriptions for the prompt ------------------------------------
    def _tool_descriptions(self) -> str:
        lines = []
        for t in self.tools.values():
            params = ", ".join(f"{k}: {v.get('type','any')}" for k, v in t.parameters.items())
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines) or "(no tools available)"

    # -- Parsing helpers -----------------------------------------------------
    @staticmethod
    def _parse_action(text: str):
        """Extract tool name and JSON args from 'Action: tool_name({...})'."""
        match = re.search(r"Action:\s*(\w+)\((.+)\)\s*$", text, re.DOTALL | re.MULTILINE)
        if not match:
            return None, None
        name = match.group(1)
        raw_args = match.group(2).strip()
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            # Try wrapping bare strings
            if ":" not in raw_args:
                args = {"expression": raw_args} if name == "calculator" else {"query": raw_args}
            else:
                args = {"query": raw_args}
        return name, args

    # -- Execute a tool with optional human confirmation ---------------------
    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self.tools:
            return f"Error: unknown tool '{name}'. Available: {list(self.tools.keys())}"

        if self.human_in_loop:
            print(f"\n  >> Tool call: {name}({json.dumps(args)})")
            confirm = input("  >> Approve? [Y/n]: ").strip().lower()
            if confirm and confirm != "y":
                return "Tool execution cancelled by user."

        tool = self.tools[name]
        try:
            return tool.callable_fn(**args)
        except TypeError as exc:
            return f"Error calling {name}: {exc}"

    # -- Main agent loop -----------------------------------------------------
    def run(self, query: str) -> AgentResult:
        """Execute the ReAct loop for a user query."""
        react_sys = self.REACT_INSTRUCTION.format(tool_descriptions=self._tool_descriptions())
        full_system = f"{self.system_prompt}\n\n{react_sys}"

        self.memory.add("user", query)
        reasoning_trace: List[str] = []
        tools_used: List[str] = []

        for iteration in range(1, self.max_iterations + 1):
            messages = [{"role": "system", "content": full_system}] + self.memory.get_context()
            llm_output = self._call_llm(messages)
            reasoning_trace.append(f"[Iter {iteration}] {llm_output}")
            logger.info(f"Iteration {iteration}: {llm_output[:120]}...")

            # Check for final answer
            answer_match = re.search(r"Answer:\s*(.+)", llm_output, re.DOTALL)
            if answer_match and "Action:" not in llm_output:
                final = answer_match.group(1).strip()
                self.memory.add("assistant", final)
                return AgentResult(answer=final, reasoning_trace=reasoning_trace,
                                   tools_used=tools_used, iterations=iteration)

            # Check for tool action
            tool_name, tool_args = self._parse_action(llm_output)
            if tool_name:
                self.memory.add("assistant", llm_output)
                observation = self._execute_tool(tool_name, tool_args)
                tools_used.append(tool_name)
                reasoning_trace.append(f"[Observation] {observation[:300]}")
                self.memory.add("user", f"Observation: {observation}")
            else:
                # LLM produced a thought but no action/answer -- nudge it
                self.memory.add("assistant", llm_output)
                self.memory.add("user", "Continue with an Action or provide the final Answer.")

        return AgentResult(
            answer="Max iterations reached without a final answer.",
            reasoning_trace=reasoning_trace, tools_used=tools_used, iterations=self.max_iterations,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_tools(tool_names: List[str]) -> List[ToolDefinition]:
    """Map CLI tool names to ToolDefinition objects."""
    if "all" in tool_names:
        return list(BUILTIN_TOOLS.values())
    resolved = []
    for name in tool_names:
        name = name.strip()
        if name in BUILTIN_TOOLS:
            resolved.append(BUILTIN_TOOLS[name])
        else:
            logger.warning(f"Unknown tool '{name}', skipping. Available: {list(BUILTIN_TOOLS.keys())}")
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Tool-using LLM agent builder (ReAct)")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.",
                        help="System prompt for the agent")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--tools", default="all",
                        help="Comma-separated tool list: calculator,python,file,web_search,all")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max ReAct loop iterations")
    parser.add_argument("--human-in-loop", action="store_true",
                        help="Require human confirmation before each tool call")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--verbose", action="store_true", help="Print full reasoning trace")
    args = parser.parse_args()

    tool_list = resolve_tools(args.tools.split(","))
    logger.info(f"Agent initialized: model={args.model}, tools={[t.name for t in tool_list]}, "
                f"max_iter={args.max_iterations}, human_in_loop={args.human_in_loop}")

    agent = Agent(
        system_prompt=args.system_prompt,
        tools=tool_list,
        model=args.model,
        max_iterations=args.max_iterations,
        human_in_loop=args.human_in_loop,
        api_base=args.api_base,
    )

    print("\nAgent ready. Type your query (or 'quit' to exit).\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        result = agent.run(query)
        print(f"\nAgent: {result.answer}")
        print(f"  [iterations={result.iterations}, tools_used={result.tools_used}]")
        if args.verbose:
            print("\n--- Reasoning Trace ---")
            for step in result.reasoning_trace:
                print(f"  {step}")
            print("--- End Trace ---\n")


if __name__ == "__main__":
    main()
