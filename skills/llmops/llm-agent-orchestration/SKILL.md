---
name: llm-agent-orchestration
description: >
  Build and orchestrate LLM-powered agents. Covers agent architectures (ReAct, Plan-and-Execute, multi-agent),
  tool use and function calling, LangChain agents, LangGraph workflows, CrewAI multi-agent systems, Autogen,
  agent memory (short-term, long-term, episodic), agent planning and reasoning, error handling and recovery,
  agent evaluation and testing, human-in-the-loop patterns, agent observability, production agent deployment,
  and agent safety guardrails. Use when building LLM agents, implementing tool use, orchestrating multi-agent
  systems, or deploying agents to production.
license: Apache-2.0
metadata:
  author: llmops-skills
  version: "1.0"
  category: llmops
---

# LLM Agent Orchestration

## Overview

LLM agents extend language models with the ability to reason, plan, use tools, and take
actions. Orchestration manages complex multi-step workflows and multi-agent collaboration.

Tested with: `langgraph>=1.0`, `langchain>=1.0`, `crewai>=1.0`, `openai>=1.0`.

## When to Use This Skill

- Building LLM-powered agents with tool use
- Orchestrating multi-step reasoning workflows
- Setting up multi-agent collaboration
- Adding memory to agent conversations
- Deploying agents safely to production

## Agent Architecture Patterns

```
ReAct Agent:                    Plan-and-Execute:
Think → Act → Observe → ...     Plan → Execute Step 1 → ... → Replan

Multi-Agent:                    Hierarchical:
Agent A ←→ Agent B              Supervisor
    ↕           ↕                 ├─ Worker Agent 1
Agent C ←→ Agent D                ├─ Worker Agent 2
                                  └─ Worker Agent 3
```

## Step-by-Step Instructions

### 1. Tool-Using Agent with OpenAI

```python
from openai import OpenAI
import json

client = OpenAI()

# strict: True + additionalProperties: false enables structured-output
# validation of tool arguments. Under strict mode every property must be listed
# in "required", and JSON Schema "default" is NOT honored - the model either
# supplies a value or you apply the default yourself after parsing.
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the product database for items matching a query",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "enum": ["electronics", "clothing", "books"]},
                    "max_results": {"type": "integer", "description": "Max results, use 5 if unspecified"},
                },
                "required": ["query", "category", "max_results"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
]

def run_agent(user_message, max_iterations=5):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": user_message},
    ]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-5-mini", messages=messages, tools=tools
        )

        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "length":
            return "Response truncated - raise max_tokens or shorten the context"

        if message.tool_calls:
            # Append the assistant turn as a plain dict; passing the SDK object
            # back works today but model_dump() is the documented round-trip.
            messages.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                result = execute_tool(tool_call.function.name,
                                     json.loads(tool_call.function.arguments))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })
        else:
            return message.content

    return "Max iterations reached"
```

### 2. LangGraph Workflow

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator

llm = ChatOpenAI(model="gpt-5-mini")

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str

def researcher(state: AgentState):
    """Research step - gather information."""
    messages = state["messages"]
    response = llm.invoke(
        [{"role": "system", "content": "Research the topic thoroughly."}]
        + messages
    )
    return {"messages": [response], "next_action": "analyze"}

def analyzer(state: AgentState):
    """Analysis step - analyze gathered information."""
    response = llm.invoke(
        [{"role": "system", "content": "Analyze the research findings."}]
        + state["messages"]
    )
    return {"messages": [response], "next_action": "write"}

def writer(state: AgentState):
    """Writing step - produce final output."""
    response = llm.invoke(
        [{"role": "system", "content": "Write a clear summary based on the analysis."}]
        + state["messages"]
    )
    return {"messages": [response], "next_action": "end"}

# Build graph. This pipeline is strictly linear, so use plain edges: a
# conditional edge with a single destination adds a routing hop for nothing.
# Reserve add_conditional_edges for real branches.
workflow = StateGraph(AgentState)
workflow.add_node("research", researcher)
workflow.add_node("analyze", analyzer)
workflow.add_node("write", writer)

workflow.add_edge(START, "research")   # modern idiom; set_entry_point still works
workflow.add_edge("research", "analyze")
workflow.add_edge("analyze", "write")
workflow.add_edge("write", END)

app = workflow.compile()
result = app.invoke({"messages": [{"role": "user", "content": "Research MLOps trends"}]})
```

For a plain tool-calling agent, do not hand-build the graph. LangGraph's
`langgraph.prebuilt.create_react_agent` is deprecated in favour of
`from langchain.agents import create_agent`, which takes a model string,
middleware, and a checkpointer:

```python
from langchain.agents import create_agent

agent = create_agent("openai:gpt-5-mini", tools=[search_database, get_weather])
result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Tokyo?"}]})
```

### 3. Multi-Agent with CrewAI

```python
from crewai import Agent, Task, Crew, Process

# Define agents
researcher = Agent(
    role="ML Research Analyst",
    goal="Research the latest MLOps tools and best practices",
    backstory="Expert ML engineer with deep knowledge of MLOps ecosystem",
    tools=[search_tool, web_scraper],
    llm="gpt-5-mini",
)

architect = Agent(
    role="ML Systems Architect",
    goal="Design scalable ML pipeline architectures",
    backstory="Senior architect who has designed ML platforms at scale",
    tools=[diagram_tool],
    llm="gpt-5-mini",
)

writer = Agent(
    role="Technical Writer",
    goal="Create clear technical documentation",
    backstory="Experienced technical writer specialized in ML documentation",
    llm="gpt-5-mini",
)

# Define tasks
research_task = Task(
    description="Research the current state of MLOps tooling for {topic}",
    expected_output="Comprehensive research report with tool comparisons",
    agent=researcher,
)

design_task = Task(
    description="Based on the research, design an architecture for {topic}",
    expected_output="Architecture document with diagrams",
    agent=architect,
)

doc_task = Task(
    description="Write user-facing documentation based on the architecture",
    expected_output="Clear, actionable documentation",
    agent=writer,
)

# Run crew
crew = Crew(
    agents=[researcher, architect, writer],
    tasks=[research_task, design_task, doc_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff(inputs={"topic": "feature store implementation"})
```

### 4. Agent Memory

```python
class AgentMemory:
    def __init__(self, max_short_term=20, embeddings_model=None):
        self.short_term = []       # Recent conversation
        self.long_term = []        # Persistent knowledge
        self.max_short_term = max_short_term
        self.embeddings = embeddings_model

    def add_to_short_term(self, message):
        self.short_term.append(message)
        if len(self.short_term) > self.max_short_term:
            # Summarize and move to long-term
            summary = self.summarize(self.short_term[:5])
            self.long_term.append(summary)
            self.short_term = self.short_term[5:]

    def retrieve_relevant(self, query, top_k=3):
        """Retrieve relevant memories for current context."""
        if not self.embeddings or not self.long_term:
            return []
        query_embedding = self.embeddings.encode(query)
        scored = []
        for memory in self.long_term:
            mem_embedding = self.embeddings.encode(memory["content"])
            score = cosine_similarity(query_embedding, mem_embedding)
            scored.append((memory, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, s in scored[:top_k]]

    def build_context(self, current_query):
        """Build full context for the agent."""
        relevant = self.retrieve_relevant(current_query)
        return {
            "short_term": self.short_term[-10:],
            "relevant_memories": relevant,
        }
```

### 5. Human-in-the-Loop

```python
class HumanInTheLoopAgent:
    def __init__(self, agent, approval_required_tools=None):
        self.agent = agent
        self.approval_required = approval_required_tools or ["execute_code", "send_email"]

    def run(self, query):
        plan = self.agent.plan(query)

        for step in plan:
            if step.tool in self.approval_required:
                approved = self.request_approval(step)
                if not approved:
                    return "Action cancelled by user"

            result = self.agent.execute_step(step)

            # Check if result needs human review
            if result.confidence < 0.7:
                human_feedback = self.request_review(step, result)
                result = self.agent.refine(result, human_feedback)

        return result
```

## Best Practices

1. **Limit tool access** - Only give agents the tools they need
2. **Never fake a sandbox** - `eval`/`exec` with a stripped `__builtins__` and a
   blocklist of forbidden substrings is not isolation; both are escapable in one
   line. Use an AST whitelist for arithmetic (see `scripts/build_agent.py`) and a
   container or hosted code-interpreter service for real code execution.
3. **Set max iterations** - Prevent infinite loops
4. **Implement timeouts** - Agents can get stuck
5. **Log all actions** for debugging and audit
6. **Use structured outputs** for tool arguments (`strict: True`)
7. **Human-in-the-loop** for high-stakes actions
8. **Test with adversarial inputs** - Agents can be manipulated
9. **Monitor token usage** - Agents can be expensive
10. **Use checkpoints** for long-running multi-step tasks

## Scripts

- `scripts/build_agent.py` - Agent construction framework
- `scripts/multi_agent.py` - Multi-agent orchestration setup

## References

See [references/REFERENCE.md](references/REFERENCE.md) for framework comparisons.

## Related skills

**Upstream:** `llm-deployment` (the model endpoint agents call — this skill is the application layer on top) · **Downstream:** `llm-observability` (agent/tool traces) and `llm-guardrails` (tool-use safety)
**See also:** `llm-prompt-engineering` for system prompts and tool descriptions · `llm-evaluation` for end-to-end agent evals
