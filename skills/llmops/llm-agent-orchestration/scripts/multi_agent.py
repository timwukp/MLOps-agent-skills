#!/usr/bin/env python3
"""Multi-agent orchestration with sequential, parallel, and hierarchical patterns.

Usage:
    python multi_agent.py --pattern sequential --task "Research and write about MLOps trends"
    python multi_agent.py --pattern parallel --task "Analyze this dataset from three angles"
    python multi_agent.py --pattern hierarchical --task "Build a comprehensive project plan"
    python multi_agent.py --config agents.yaml --pattern sequential --task "..." --output report.json
"""
import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    QUESTION = "question"
    DELEGATION = "delegation"


@dataclass
class Message:
    """Structured message passed between agents."""
    sender: str
    receiver: str
    content: str
    msg_type: MessageType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class AgentConfig:
    """Configuration for a single agent in the orchestration."""
    name: str
    role: str
    system_prompt: str
    model: str = "gpt-5-mini"
    tools: List[str] = field(default_factory=list)
    can_delegate_to: List[str] = field(default_factory=list)


@dataclass
class ExecutionTrace:
    """Full trace of an orchestration run for debugging."""
    messages: List[Message] = field(default_factory=list)
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    pattern: str = ""
    task: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def log_message(self, msg: Message):
        self.messages.append(msg)
        logger.info(f"[{msg.msg_type.value}] {msg.sender} -> {msg.receiver}: {msg.content[:100]}...")

    def to_dict(self) -> Dict[str, Any]:
        self.end_time = time.time()
        return {
            "pattern": self.pattern,
            "task": self.task,
            "duration_seconds": round((self.end_time or 0) - self.start_time, 2),
            "num_messages": len(self.messages),
            "messages": [
                {"sender": m.sender, "receiver": m.receiver, "type": m.msg_type.value,
                 "content": m.content[:500], "timestamp": m.timestamp}
                for m in self.messages
            ],
            "agent_outputs": self.agent_outputs,
        }


# ---------------------------------------------------------------------------
# Single agent wrapper
# ---------------------------------------------------------------------------

class AgentRunner:
    """Thin wrapper that calls an OpenAI-compatible LLM for a single agent."""

    def __init__(self, config: AgentConfig, api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.config = config
        # OPENAI_BASE_URL is the openai>=1.0 variable (OPENAI_API_BASE was v0).
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or None
        self._client = None

    def _get_client(self):
        """Create the client once; it is thread-safe and pools connections."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                logger.error("Install: pip install openai")
                sys.exit(1)
            self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        return self._client

    def invoke(self, task_content: str, context: str = "") -> str:
        """Send a task to the agent and return its textual response."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
        ]
        if context:
            messages.append({"role": "user", "content": f"Context from previous steps:\n{context}"})
        messages.append({"role": "user", "content": task_content})

        response = self._get_client().chat.completions.create(
            model=self.config.model, messages=messages, temperature=0.3, max_tokens=1500,
        )
        return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Task routing
# ---------------------------------------------------------------------------

KEYWORD_ROUTES: Dict[str, List[str]] = {
    "research": ["research", "find", "search", "investigate", "look up", "gather"],
    "writing": ["write", "draft", "compose", "create", "summarize", "document"],
    "review": ["review", "check", "verify", "evaluate", "critique", "proofread"],
    "code": ["code", "implement", "program", "develop", "build", "debug"],
    "analysis": ["analyze", "analyse", "compare", "measure", "assess", "data"],
}


def keyword_route(task: str, agents: Dict[str, AgentConfig]) -> Optional[str]:
    """Route a task to a specialist agent based on keyword matching."""
    task_lower = task.lower()
    scores: Dict[str, int] = {}
    for agent_name, config in agents.items():
        role_key = config.role.lower()
        score = 0
        for category, keywords in KEYWORD_ROUTES.items():
            if category in role_key:
                for kw in keywords:
                    if kw in task_lower:
                        score += 1
        scores[agent_name] = score
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Manages multiple agents and routes tasks using different patterns."""

    def __init__(self, agents: List[AgentConfig], api_base: Optional[str] = None):
        self.agents: Dict[str, AgentConfig] = {a.name: a for a in agents}
        self.runners: Dict[str, AgentRunner] = {
            a.name: AgentRunner(a, api_base=api_base) for a in agents
        }
        self.trace = ExecutionTrace()

    # -- Sequential pipeline ------------------------------------------------
    def run_sequential(self, task: str, agent_order: Optional[List[str]] = None) -> str:
        """Pipeline: each agent passes its output as context to the next."""
        self.trace.pattern = "sequential"
        self.trace.task = task
        order = agent_order or list(self.agents.keys())
        context = ""

        for agent_name in order:
            if agent_name not in self.runners:
                logger.warning(f"Agent '{agent_name}' not found, skipping.")
                continue
            msg_in = Message(sender="orchestrator", receiver=agent_name,
                             content=task, msg_type=MessageType.TASK)
            self.trace.log_message(msg_in)

            output = self.runners[agent_name].invoke(task, context=context)
            self.trace.agent_outputs[agent_name] = output

            msg_out = Message(sender=agent_name, receiver="orchestrator",
                              content=output, msg_type=MessageType.RESULT)
            self.trace.log_message(msg_out)
            context = f"{context}\n\n--- {agent_name} output ---\n{output}" if context else output

        return context

    # -- Parallel fan-out / fan-in ------------------------------------------
    def run_parallel(self, task: str, agent_names: Optional[List[str]] = None,
                     max_workers: Optional[int] = None) -> str:
        """Fan-out: all agents work on the same task concurrently, then results merge.

        LLM calls are I/O-bound, so a thread pool gives real concurrency despite
        the GIL: wall-clock time is that of the slowest agent, not their sum.
        """
        self.trace.pattern = "parallel"
        self.trace.task = task
        names = [n for n in (agent_names or list(self.agents.keys())) if n in self.runners]
        results: Dict[str, str] = {}

        for agent_name in names:
            self.trace.log_message(Message(sender="orchestrator", receiver=agent_name,
                                           content=task, msg_type=MessageType.TASK))

        with ThreadPoolExecutor(max_workers=max_workers or max(len(names), 1)) as pool:
            futures = {pool.submit(self.runners[n].invoke, task): n for n in names}
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    output = future.result()
                except Exception as exc:
                    logger.error(f"Agent '{agent_name}' failed: {exc}")
                    output = f"[error: {type(exc).__name__}: {exc}]"
                results[agent_name] = output
                self.trace.agent_outputs[agent_name] = output
                self.trace.log_message(Message(sender=agent_name, receiver="orchestrator",
                                               content=output, msg_type=MessageType.RESULT))

        # Fan-in: combine results in the requested order, not completion order.
        combined_parts = []
        for name in names:
            role = self.agents[name].role
            combined_parts.append(f"## {name} ({role})\n{results.get(name, '')}")
        return "\n\n".join(combined_parts)

    # -- Hierarchical delegation -------------------------------------------
    def run_hierarchical(self, task: str, manager_name: Optional[str] = None) -> str:
        """Manager agent delegates sub-tasks to worker agents."""
        self.trace.pattern = "hierarchical"
        self.trace.task = task

        # Default manager: the first agent that actually declares delegation
        # targets. Falling back to "first agent in the dict" picked a worker in
        # the default pipeline (researcher), where the reviewer is the manager.
        if manager_name:
            manager = manager_name
        else:
            manager = next((n for n, c in self.agents.items() if c.can_delegate_to),
                           next(iter(self.agents)))
            logger.info(f"No manager specified; using '{manager}'")
        if manager not in self.agents:
            raise ValueError(f"Manager '{manager}' not in agents: {list(self.agents)}")
        manager_config = self.agents[manager]
        worker_names = manager_config.can_delegate_to or [
            n for n in self.agents if n != manager
        ]

        # Step 1: Manager breaks down the task
        worker_descriptions = "\n".join(
            f"- {n}: {self.agents[n].role}" for n in worker_names if n in self.agents
        )
        planning_prompt = (
            f"You are the manager. Break the following task into sub-tasks and assign each to "
            f"one of your workers. Reply with a JSON list of objects with keys 'worker' and 'subtask'.\n\n"
            f"Available workers:\n{worker_descriptions}\n\nTask: {task}"
        )
        msg_plan = Message(sender="orchestrator", receiver=manager,
                           content=planning_prompt, msg_type=MessageType.TASK)
        self.trace.log_message(msg_plan)

        plan_output = self.runners[manager].invoke(planning_prompt)
        self.trace.agent_outputs[f"{manager}_plan"] = plan_output
        logger.info(f"Manager plan: {plan_output[:200]}...")

        # Parse sub-tasks from JSON
        subtasks = self._parse_subtasks(plan_output, worker_names)

        # Step 2: Workers execute their sub-tasks. A worker can receive more than
        # one sub-task, so results accumulate in a list per worker rather than
        # overwriting the previous output.
        worker_results: Dict[str, List[str]] = {}
        for idx, item in enumerate(subtasks, 1):
            worker = item.get("worker", "")
            subtask = item.get("subtask", "")
            if not subtask:
                logger.warning(f"Sub-task {idx} has no 'subtask' text, skipping.")
                continue
            if worker not in self.runners:
                logger.warning(f"Worker '{worker}' not found, routing via keywords.")
                worker = keyword_route(subtask, self.agents) or next(iter(worker_names), None)
                if not worker:
                    continue

            msg_del = Message(sender=manager, receiver=worker,
                              content=subtask, msg_type=MessageType.DELEGATION)
            self.trace.log_message(msg_del)

            output = self.runners[worker].invoke(subtask)
            worker_results.setdefault(worker, []).append(output)
            self.trace.agent_outputs[f"{worker}_{len(worker_results[worker])}"] = output

            msg_res = Message(sender=worker, receiver=manager,
                              content=output, msg_type=MessageType.RESULT)
            self.trace.log_message(msg_res)

        # Step 3: Manager synthesizes
        synthesis_context = "\n\n".join(
            f"[{w} #{i}]: {r}"
            for w, outs in worker_results.items()
            for i, r in enumerate(outs, 1)
        )
        synthesis_prompt = (
            f"You received these results from your workers. Synthesize them into a "
            f"coherent final response for the original task: {task}"
        )
        final = self.runners[manager].invoke(synthesis_prompt, context=synthesis_context)
        self.trace.agent_outputs[f"{manager}_final"] = final
        return final

    @staticmethod
    def _parse_subtasks(text: str, valid_workers: List[str]) -> List[Dict[str, str]]:
        """Extract sub-task assignments from the manager's JSON plan.

        Fails loudly instead of falling back to "give the whole raw plan to every
        worker", which silently multiplied cost by len(workers) and produced
        duplicated work that looked like a successful run.
        """
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            raise ValueError(
                "Manager plan contained no JSON array of sub-tasks. Raw output: "
                f"{text[:300]!r}")
        try:
            plan = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Manager plan is not valid JSON ({exc}). "
                             f"Raw output: {text[:300]!r}") from exc
        if not isinstance(plan, list):
            raise ValueError(f"Manager plan must be a JSON list, got {type(plan).__name__}")
        items = [it for it in plan if isinstance(it, dict) and it.get("subtask")]
        if not items:
            raise ValueError(f"Manager plan had no usable {{worker, subtask}} entries: "
                             f"{text[:300]!r}")
        unknown = {it.get("worker") for it in items} - set(valid_workers)
        if unknown:
            logger.warning(f"Plan references unknown workers {unknown}; "
                           f"they will be keyword-routed.")
        return items


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> List[AgentConfig]:
    """Load agent definitions from a YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    raw = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            logger.error("Install: pip install pyyaml")
            sys.exit(1)
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    agents_raw = data if isinstance(data, list) else data.get("agents", [])
    agents = []
    for item in agents_raw:
        agents.append(AgentConfig(
            name=item["name"],
            role=item.get("role", item["name"]),
            system_prompt=item.get("system_prompt", f"You are a {item.get('role', 'assistant')}."),
            model=item.get("model", "gpt-5-mini"),
            tools=item.get("tools", []),
            can_delegate_to=item.get("can_delegate_to", []),
        ))
    logger.info(f"Loaded {len(agents)} agents from {config_path}")
    return agents


def default_pipeline_agents() -> List[AgentConfig]:
    """Return a default three-agent research/write/review pipeline."""
    return [
        AgentConfig(
            name="researcher",
            role="research",
            system_prompt=(
                "You are a thorough research analyst. Gather key facts, data points, "
                "and insights on the given topic. Be specific and cite sources where possible."
            ),
        ),
        AgentConfig(
            name="writer",
            role="writing",
            system_prompt=(
                "You are a skilled technical writer. Using the research provided, produce "
                "a clear, well-structured document. Use headings and bullet points."
            ),
        ),
        AgentConfig(
            name="reviewer",
            role="review",
            system_prompt=(
                "You are a meticulous editor and reviewer. Check the document for accuracy, "
                "clarity, completeness, and style. Provide the final polished version."
            ),
            can_delegate_to=["researcher", "writer"],
        ),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-agent orchestration")
    parser.add_argument("--config", default=None,
                        help="YAML/JSON file with agent definitions")
    parser.add_argument("--pattern", choices=["sequential", "parallel", "hierarchical"],
                        default="sequential", help="Orchestration pattern")
    parser.add_argument("--task", required=True, help="Task description for the agents")
    parser.add_argument("--output", default=None,
                        help="Path to write JSON output with trace")
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--verbose", action="store_true", help="Print full execution trace")
    args = parser.parse_args()

    # Load agents
    if args.config:
        agents = load_config(args.config)
    else:
        agents = default_pipeline_agents()
        logger.info("Using default research -> writing -> review pipeline.")

    orchestrator = Orchestrator(agents, api_base=args.api_base)
    logger.info(f"Running pattern={args.pattern} with {len(agents)} agents")

    # Execute
    try:
        if args.pattern == "sequential":
            result = orchestrator.run_sequential(args.task)
        elif args.pattern == "parallel":
            result = orchestrator.run_parallel(args.task)
        else:
            result = orchestrator.run_hierarchical(args.task)
    except ValueError as exc:
        logger.error(f"Orchestration failed: {exc}")
        sys.exit(1)

    # Output
    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(result)

    if args.output:
        trace_data = orchestrator.trace.to_dict()
        trace_data["final_output"] = result
        Path(args.output).write_text(json.dumps(trace_data, indent=2, default=str),
                                     encoding="utf-8")
        logger.info(f"Trace written to {args.output}")

    if args.verbose:
        print("\n--- Execution Trace ---")
        for msg in orchestrator.trace.messages:
            print(f"  [{msg.msg_type.value}] {msg.sender} -> {msg.receiver}: "
                  f"{msg.content[:120]}...")
        print(f"--- {len(orchestrator.trace.messages)} messages total ---")


if __name__ == "__main__":
    main()
