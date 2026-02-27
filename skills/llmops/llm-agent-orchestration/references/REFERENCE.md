# LLM Agent Orchestration Reference

## Agent Frameworks Comparison

| Feature | LangGraph | CrewAI | AutoGen | Semantic Kernel | Haystack |
|---|---|---|---|---|---|
| **License** | MIT | MIT | MIT (Creative Commons) | MIT | Apache 2.0 |
| **Language** | Python, JS | Python | Python, .NET | Python, C#, Java | Python |
| **Architecture** | Graph-based state machine | Role-based crew | Conversational agents | Plugin-based kernel | Pipeline-based |
| **Multi-Agent** | Yes (native) | Yes (primary focus) | Yes (primary focus) | Yes (via planners) | Limited |
| **Human-in-Loop** | Native support | Callback-based | Native support | Native support | Custom nodes |
| **Streaming** | Yes | Limited | Yes | Yes | Yes |
| **Persistence** | Built-in checkpointing | Limited | Limited | Custom | Custom |
| **Tool Calling** | LangChain tools | Custom tools | Function calling | Plugins/functions | Custom components |
| **State Management** | Explicit typed state | Implicit via tasks | Conversation history | Kernel memory | Pipeline state |
| **Learning Curve** | Medium (graph concepts) | Low (declarative) | Medium | Medium-High | Medium |
| **Production Readiness** | High | Medium | Medium | High (Microsoft-backed) | High |
| **Best For** | Complex workflows, cycles | Team-style collaboration | Research, multi-turn dialog | Enterprise .NET/Java | RAG-heavy pipelines |

## Agent Patterns

### ReAct (Reasoning + Acting)

```python
# ReAct loop: Think -> Act -> Observe -> Repeat
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model="gpt-4o")
tools = [search_tool, calculator_tool, lookup_tool]

agent = create_react_agent(llm, tools)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the GDP per capita of the country with the tallest building?"}]
})
```

ReAct flow:
```
Thought: I need to find the country with the tallest building.
Action: search("tallest building in the world")
Observation: The Burj Khalifa in Dubai, UAE is the tallest building at 828m.
Thought: Now I need to find the GDP per capita of the UAE.
Action: search("UAE GDP per capita 2025")
Observation: The UAE GDP per capita is approximately $53,000 USD.
Thought: I have the answer.
Final Answer: The GDP per capita of the UAE is approximately $53,000 USD.
```

### Plan-and-Execute

```python
# Plan-and-Execute: Create full plan first, then execute steps
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
import operator

class PlanState(TypedDict):
    objective: str
    plan: List[str]
    completed_steps: Annotated[list, operator.add]
    current_step: int
    results: Annotated[list, operator.add]
    final_answer: str

def planner(state: PlanState) -> dict:
    """Generate a step-by-step plan for the objective."""
    plan = llm.invoke(f"Create a plan to: {state['objective']}")
    return {"plan": parse_steps(plan), "current_step": 0}

def executor(state: PlanState) -> dict:
    """Execute the current step of the plan."""
    step = state["plan"][state["current_step"]]
    result = agent.invoke(step)
    return {
        "results": [result],
        "completed_steps": [step],
        "current_step": state["current_step"] + 1
    }

def should_continue(state: PlanState) -> str:
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "execute"
```

### LATS (Language Agent Tree Search)

Key concepts:
- Explore multiple reasoning paths simultaneously using tree search.
- Score candidate actions using value functions or self-evaluation.
- Backtrack when paths score poorly, similar to Monte Carlo Tree Search.
- Best for complex, multi-step problems where greedy approaches fail.

### Reflexion

Key concepts:
- Agent attempts a task and receives feedback on its performance.
- Agent reflects on failures, generating verbal self-critique.
- Reflection is stored in memory and used to improve subsequent attempts.
- Implements a learning loop within a single task execution.

```
Attempt 1 --> Evaluate --> Reflect ("I made an error because I didn't verify...")
Attempt 2 (with reflection context) --> Evaluate --> Success
```

## Tool Use Patterns

### Function Calling

```python
# OpenAI-style function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)
```

### Code Execution (Sandboxed)

```python
# Safe code execution for agent tool use
import subprocess
import tempfile

def execute_python_code(code: str, timeout: int = 30) -> str:
    """Execute Python code in an isolated subprocess."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ["python", f.name],
                capture_output=True, text=True, timeout=timeout,
                env={"PATH": "/usr/bin"}  # Restricted environment
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out"
```

### API Integration Best Practices

- Always provide clear, concise tool descriptions; the LLM relies on these to select tools.
- Include parameter descriptions with examples of valid values.
- Return structured responses from tools, not raw API payloads.
- Implement timeout and retry logic within tool functions.
- Log every tool invocation for debugging.
- Limit the number of tools available (under 20 is ideal; too many confuse the model).

## Memory Systems

### Short-Term Memory (Buffer)

```python
# Conversation buffer with sliding window
class ConversationBuffer:
    def __init__(self, max_tokens=4000):
        self.messages = []
        self.max_tokens = max_tokens

    def add(self, message):
        self.messages.append(message)
        while self._total_tokens() > self.max_tokens:
            self.messages.pop(0)  # Drop oldest messages

    def get_context(self):
        return self.messages
```

### Long-Term Memory (Vector Store)

```python
# Semantic memory using vector store
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

memory_store = Chroma(
    collection_name="agent_memory",
    embedding_function=OpenAIEmbeddings()
)

# Store a memory
memory_store.add_texts(
    texts=["User prefers concise responses"],
    metadatas=[{"type": "preference", "user_id": "u123", "timestamp": "2026-02-27"}]
)

# Retrieve relevant memories
relevant = memory_store.similarity_search(
    query="How should I format this response?",
    k=5, filter={"user_id": "u123"}
)
```

### Episodic Memory

```python
# Episodic memory: store and retrieve past task experiences
class EpisodicMemory:
    def __init__(self, vector_store):
        self.store = vector_store

    def record_episode(self, task, plan, outcome, reflection):
        episode = {
            "task": task,
            "plan": plan,
            "outcome": outcome,
            "reflection": reflection,
            "timestamp": datetime.now().isoformat()
        }
        self.store.add_texts(
            texts=[f"Task: {task}\nOutcome: {outcome}\nLesson: {reflection}"],
            metadatas=[episode]
        )

    def recall_similar(self, current_task, k=3):
        return self.store.similarity_search(current_task, k=k)
```

## Multi-Agent Architectures

### Hierarchical (Supervisor Pattern)

```
        Supervisor Agent
       /       |        \
  Researcher  Writer   Reviewer
```

- A supervisor agent delegates tasks to specialized sub-agents.
- The supervisor decides which agent to invoke and when to synthesize results.
- Best for well-defined workflows with clear role boundaries.

### Collaborative (Peer Pattern)

```
  Agent A <--> Agent B
    ^             ^
    |             |
    v             v
  Agent C <--> Agent D
```

- Agents communicate as peers, sharing information and coordinating.
- No single point of control; consensus-driven or round-robin.
- Best for brainstorming, debate, or multi-perspective analysis.

### Competitive (Debate Pattern)

```
  Proposer --> Critic --> Judge
      ^                    |
      |____________________|
```

- One agent proposes, another critiques, a judge evaluates.
- Improves output quality through adversarial refinement.
- Higher token cost but better accuracy for high-stakes tasks.

## Human-in-the-Loop Patterns

```python
# LangGraph human-in-the-loop with interrupt
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(AgentState)
# ... define nodes ...

# Add interrupt before sensitive actions
graph.add_node("sensitive_action", sensitive_tool)

checkpointer = MemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["sensitive_action"]  # Pause here for approval
)

# First run - pauses at interrupt
result = app.invoke(input, config={"configurable": {"thread_id": "t1"}})

# Human reviews and approves
# Resume execution
result = app.invoke(None, config={"configurable": {"thread_id": "t1"}})
```

### When to Use Human-in-the-Loop

| Scenario | Pattern | Example |
|---|---|---|
| Irreversible actions | Interrupt before execution | Database deletion, email sending |
| Low-confidence decisions | Confidence threshold gate | Agent unsure which tool to use |
| Sensitive data access | Approval required | Accessing customer PII |
| Quality-critical output | Review before delivery | Legal or medical content |
| Learning from corrections | Feedback loop | Agent makes repeated errors |

## Agent Evaluation and Debugging

### Evaluation Dimensions

| Dimension | Metric | How to Measure |
|---|---|---|
| **Task Completion** | Success rate | Did the agent achieve the stated objective? |
| **Efficiency** | Steps to completion | Number of LLM calls and tool invocations |
| **Cost** | Total tokens used | Sum across all agent steps |
| **Latency** | End-to-end time | Wall-clock time from input to final output |
| **Tool Accuracy** | Correct tool selection rate | Was the right tool chosen at each step? |
| **Reasoning Quality** | Intermediate step correctness | Are the reasoning traces logically sound? |
| **Robustness** | Performance on edge cases | How does the agent handle unexpected inputs? |

### Common Agent Failure Modes

1. **Tool selection errors**: Agent picks the wrong tool for the task.
2. **Parameter hallucination**: Agent fabricates tool arguments not grounded in context.
3. **Infinite loops**: Agent repeats the same action without making progress.
4. **Premature termination**: Agent stops before completing the task.
5. **Context loss**: Agent forgets information from earlier in a long conversation.
6. **Over-planning**: Agent generates elaborate plans but fails to execute them.

### Debugging Strategies

- Trace every agent step (thought, action, observation) with full payloads.
- Set maximum iteration limits to prevent infinite loops (typically 10-25 steps).
- Log the full prompt sent to the LLM at each step, including system message and tools.
- Use deterministic settings (temperature=0) when reproducing issues.
- Build regression test suites from observed production failures.

## Best Practices

1. **Start simple**: Begin with ReAct; add complexity (planning, multi-agent) only when needed.
2. **Limit tool count**: Keep tools under 15-20 per agent; split into sub-agents if more are needed.
3. **Write excellent tool descriptions**: This is the single most impactful factor in agent accuracy.
4. **Implement guardrails**: Set max iterations, token budgets, and action allowlists.
5. **Use checkpointing**: Persist agent state so you can resume, replay, and debug.
6. **Test adversarially**: Include edge cases, ambiguous inputs, and multi-step tasks in evaluations.
7. **Monitor in production**: Track step counts, tool usage patterns, and failure rates.

## Common Pitfalls

- Building multi-agent systems when a single agent with good tools would suffice.
- Not setting maximum iteration limits, leading to runaway costs.
- Providing vague tool descriptions that lead to selection errors.
- Storing entire conversation history without summarization, hitting context limits.
- Not handling tool execution failures gracefully within the agent loop.
- Evaluating only final output, ignoring intermediate reasoning quality.

## Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [Haystack Documentation](https://docs.haystack.deepset.ai/)
- [ReAct Paper: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Reflexion Paper](https://arxiv.org/abs/2303.11366)
- [LATS Paper: Language Agent Tree Search](https://arxiv.org/abs/2310.04406)
- [Andrew Ng - Agentic Design Patterns](https://www.deeplearning.ai/the-batch/agentic-design-patterns/)
