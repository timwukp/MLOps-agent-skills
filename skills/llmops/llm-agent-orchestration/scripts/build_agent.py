#!/usr/bin/env python3
"""Tool-using LLM agent builder with ReAct-style reasoning loop.

Usage:
    python build_agent.py --system-prompt "You are a helpful assistant." --tools all
    python build_agent.py --tools calculator,file --max-iterations 5
    python build_agent.py --model gpt-5 --tools all --human-in-loop
    python build_agent.py --system-prompt "Research assistant" --tools file,calculator

Security: the python_executor tool runs arbitrary Python with your privileges and
is disabled unless --enable-python-executor is passed. See its docstring.
"""
import argparse
import ast
import json
import logging
import math
import operator
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PYTHON_EXEC_TIMEOUT_S = 10


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


# Arithmetic evaluator built on the AST, not eval(). `eval` with a stripped
# __builtins__ is NOT a sandbox: an attacker reaches arbitrary code through
# attribute traversal (e.g. ().__class__.__base__.__subclasses__()). Here only
# an explicit whitelist of node types, operators and functions can execute, so
# there is no path to attribute access, subscripting, or calls at all.
_SAFE_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_SAFE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
# math functions that take/return plain numbers, plus a few builtins.
_SAFE_FUNCTIONS = {
    name: getattr(math, name) for name in (
        "sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan", "asin",
        "acos", "atan", "atan2", "hypot", "ceil", "floor", "fabs", "factorial",
        "degrees", "radians", "gcd", "trunc", "copysign", "fmod", "pow", "dist",
    )
}
_SAFE_FUNCTIONS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
_SAFE_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}
# Guard against expressions that are cheap to write but expensive to evaluate.
MAX_EXPRESSION_LEN = 500
MAX_POW_EXPONENT = 1000


def _eval_node(node):
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"unknown name '{node.id}'")
    if isinstance(node, ast.BinOp):
        op = _SAFE_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if op is operator.pow and abs(right) > MAX_POW_EXPONENT:
            raise ValueError(f"exponent too large (>{MAX_POW_EXPONENT})")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        # Only bare `name(...)` calls; `obj.attr(...)` is rejected because the
        # func is an ast.Attribute, which is not in the whitelist.
        if not isinstance(node.func, ast.Name):
            raise ValueError("only direct function calls are allowed")
        fn = _SAFE_FUNCTIONS.get(node.func.id)
        if fn is None:
            raise ValueError(f"function '{node.func.id}' is not allowed")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return fn(*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e) for e in node.elts)
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def safe_eval_arithmetic(expression: str):
    """Evaluate an arithmetic expression without eval()/exec().

    Supports + - * / // % **, unary +/-, numeric literals, the constants
    pi/e/tau/inf and the whitelisted math functions in _SAFE_FUNCTIONS.
    Raises ValueError for anything else.
    """
    if len(expression) > MAX_EXPRESSION_LEN:
        raise ValueError(f"expression longer than {MAX_EXPRESSION_LEN} characters")
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree)


def tool_calculator(expression: str) -> str:
    """Evaluate a mathematical expression with an AST whitelist (no eval)."""
    logger.info(f"[calculator] expression={expression!r}")
    try:
        return str(safe_eval_arithmetic(expression))
    except SyntaxError as exc:
        return f"Error: invalid expression ({exc.msg})"
    except Exception as exc:
        return f"Error: {exc}"


def tool_python_executor(code: str) -> str:
    """Run a Python snippet in a subprocess. NOT A SANDBOX - see warning below.

    UNSAFE FOR UNTRUSTED INPUT. This runs arbitrary Python with the privileges of
    the current user. The subprocess boundary only limits blast radius on crash
    and enforces a wall-clock timeout; it does NOT restrict filesystem access,
    network access, or resource usage, and no blocklist of forbidden substrings
    can make it safe (imports can be spelled `__import__`, built via getattr,
    obtained from object.__subclasses__(), decoded from base64, and so on).

    Only enable this tool when the code originates from a trusted operator. For
    model-generated or user-supplied code use real isolation:
      - a container with no network, a read-only rootfs, dropped capabilities and
        cpu/memory limits (gVisor/Firecracker for a stronger boundary), or
      - a hosted code-interpreter sandbox (E2B, Modal, Daytona, or the provider's
        own code-execution tool).
    """
    logger.warning("[python_executor] executing untrusted-capable code in a subprocess; "
                   "this is NOT a security sandbox (code length=%d)", len(code))
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, timeout=PYTHON_EXEC_TIMEOUT_S,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return f"Error: execution exceeded {PYTHON_EXEC_TIMEOUT_S}s timeout"
        except OSError as exc:
            return f"Error: could not start interpreter: {exc}"
    if proc.returncode != 0:
        return f"Error (exit {proc.returncode}):\n{proc.stderr.strip()[:2000]}"
    return proc.stdout.strip()[:4000] or "(no output)"


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
        description=("Evaluate an arithmetic expression, e.g. '2**10 + sqrt(144)'. "
                     "Supports + - * / // % ** and math functions by bare name "
                     "(sqrt, log, sin, ...); pi/e/tau are available."),
        parameters={"expression": {"type": "string", "required": True}},
        callable_fn=tool_calculator,
    ),
    # UNSAFE for untrusted input - opt in with --enable-python-executor.
    "python": ToolDefinition(
        name="python_executor",
        description=("Execute a short Python snippet in a subprocess and return its stdout. "
                     "NOT SANDBOXED - runs with full user privileges."),
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
        model: str = "gpt-5-mini",
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
        # OPENAI_BASE_URL is the openai>=1.0 variable; OPENAI_API_BASE was v0.
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        # Pass None (not "") so the SDK's own env/keyring resolution still applies
        # and a missing key raises a clear error instead of a 401.
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or None
        self.memory = Memory()
        self._client = None

    # -- LLM call (OpenAI-compatible) ---------------------------------------
    def _get_client(self):
        """Create the OpenAI client once and reuse it (connection pooling)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                logger.error("Install: pip install openai")
                sys.exit(1)
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        response = self._get_client().chat.completions.create(
            model=self.model, messages=messages, temperature=0.2, max_tokens=1024,
        )
        # .content is None when the model returns only tool calls or is filtered.
        return (response.choices[0].message.content or "").strip()

    # -- Tool descriptions for the prompt ------------------------------------
    def _tool_descriptions(self) -> str:
        lines = []
        for t in self.tools.values():
            params = ", ".join(f"{k}: {v.get('type','any')}" for k, v in t.parameters.items())
            lines.append(f"- {t.name}({params}): {t.description}")
        return "\n".join(lines) or "(no tools available)"

    # -- Parsing helpers -----------------------------------------------------
    def _first_required_param(self, name: str) -> Optional[str]:
        """First required parameter of a tool, used for bare-string arguments."""
        tool = self.tools.get(name)
        if not tool:
            return None
        for pname, spec in tool.parameters.items():
            if isinstance(spec, dict) and spec.get("required"):
                return pname
        return next(iter(tool.parameters), None)

    def _parse_action(self, text: str):
        """Extract tool name and args from 'Action: tool_name({...})'.

        Non-greedy and single-line: a greedy DOTALL match would swallow every
        line after the action (including a following Thought) into the args.
        """
        match = re.search(r"^\s*Action:\s*(\w+)\((.*?)\)\s*$", text, re.MULTILINE)
        if not match:
            return None, None
        name = match.group(1)
        raw_args = match.group(2).strip()
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return name, parsed
        # Bare value (or a JSON scalar): bind it to the tool's own first required
        # parameter instead of guessing "query"/"expression".
        value = parsed if parsed is not None else raw_args.strip("\"'")
        param = self._first_required_param(name)
        if param is None:
            return name, {}
        return name, {param: value}

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
            return tool.callable_fn(**(args or {}))
        except TypeError as exc:
            return f"Error calling {name}: bad arguments ({exc})"
        except Exception as exc:
            # A failing tool must not kill the loop - feed the error back as an
            # observation so the agent can recover or report it.
            logger.warning(f"Tool '{name}' raised {type(exc).__name__}: {exc}")
            return f"Error: {name} failed with {type(exc).__name__}: {exc}"

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

UNSAFE_TOOLS = {"python"}  # require an explicit opt-in flag


def resolve_tools(tool_names: List[str], allow_unsafe: bool = False) -> List[ToolDefinition]:
    """Map CLI tool names to ToolDefinition objects.

    Tools in UNSAFE_TOOLS execute arbitrary code and are excluded unless
    allow_unsafe is set, including when "all" is requested.
    """
    requested = ([n for n in BUILTIN_TOOLS] if "all" in tool_names
                 else [n.strip() for n in tool_names])
    resolved = []
    for name in requested:
        if name not in BUILTIN_TOOLS:
            logger.warning(f"Unknown tool '{name}', skipping. Available: {list(BUILTIN_TOOLS.keys())}")
            continue
        if name in UNSAFE_TOOLS and not allow_unsafe:
            logger.warning(
                f"Tool '{name}' executes arbitrary code and is NOT sandboxed; skipping. "
                f"Pass --enable-python-executor to enable it (trusted input only).")
            continue
        resolved.append(BUILTIN_TOOLS[name])
    return resolved


def main():
    parser = argparse.ArgumentParser(description="Tool-using LLM agent builder (ReAct)")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.",
                        help="System prompt for the agent")
    parser.add_argument("--model", default="gpt-5-mini", help="LLM model name")
    parser.add_argument("--tools", default="all",
                        help="Comma-separated tool list: calculator,python,file,web_search,all")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max ReAct loop iterations")
    parser.add_argument("--human-in-loop", action="store_true",
                        help="Require human confirmation before each tool call")
    parser.add_argument("--enable-python-executor", action="store_true",
                        help="Enable the python_executor tool. It is NOT sandboxed and runs "
                             "arbitrary code as the current user - trusted input only")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--verbose", action="store_true", help="Print full reasoning trace")
    args = parser.parse_args()

    tool_list = resolve_tools(args.tools.split(","), allow_unsafe=args.enable_python_executor)
    if args.enable_python_executor:
        logger.warning("python_executor ENABLED: model-generated code will run unsandboxed. "
                       "Consider --human-in-loop and container isolation.")
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
