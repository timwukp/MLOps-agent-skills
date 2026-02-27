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

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search the product database for items matching a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "enum": ["electronics", "clothing", "books"]},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
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
            model="gpt-4o", messages=messages, tools=tools
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)
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
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

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

def router(state: AgentState):
    return state["next_action"]

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("research", researcher)
workflow.add_node("analyze", analyzer)
workflow.add_node("write", writer)

workflow.set_entry_point("research")
workflow.add_conditional_edges("research", router, {"analyze": "analyze"})
workflow.add_conditional_edges("analyze", router, {"write": "write"})
workflow.add_conditional_edges("write", router, {"end": END})

app = workflow.compile()
result = app.invoke({"messages": [{"role": "user", "content": "Research MLOps trends"}]})
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
    llm="gpt-4o",
)

architect = Agent(
    role="ML Systems Architect",
    goal="Design scalable ML pipeline architectures",
    backstory="Senior architect who has designed ML platforms at scale",
    tools=[diagram_tool],
    llm="gpt-4o",
)

writer = Agent(
    role="Technical Writer",
    goal="Create clear technical documentation",
    backstory="Experienced technical writer specialized in ML documentation",
    llm="gpt-4o",
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
2. **Set max iterations** - Prevent infinite loops
3. **Implement timeouts** - Agents can get stuck
4. **Log all actions** for debugging and audit
5. **Use structured outputs** for tool arguments
6. **Human-in-the-loop** for high-stakes actions
7. **Test with adversarial inputs** - Agents can be manipulated
8. **Monitor token usage** - Agents can be expensive
9. **Use checkpoints** for long-running multi-step tasks

## Scripts

- `scripts/build_agent.py` - Agent construction framework
- `scripts/multi_agent.py` - Multi-agent orchestration setup

## References

See [references/REFERENCE.md](references/REFERENCE.md) for framework comparisons.
