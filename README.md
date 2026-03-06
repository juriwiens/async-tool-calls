# Async Tool Calls

Experiments with **asynchronous, non-blocking tool calls** in LLM agent frameworks.

## The Problem

In most LLM agent frameworks (Google ADK, LangGraph, OpenAI Agents SDK, ...),
tool calls follow a **synchronous, blocking pattern**:

```
User: "Search for chicken products AND quick recipes"

  Main Agent
    |-- calls: search_products("chicken")     --+
    |-- calls: search_recipes("chicken")       -+  Both run in parallel,
    |                                           |  BUT the agent only sees
    |         ... waiting for BOTH ...          |  results when ALL complete
    |                                           |
    |-- receives: products (after 1s)    -------+  Has to wait
    |-- receives: recipes  (after 20s)   -------+  for the slowest
    |
    '-- finally responds to user (after 20s)
```

**The problem**: If two sub-agents are called in parallel, the main agent can
only process and stream results to the user **after the slowest one finishes**.
A fast sub-agent (1 second) is held hostage by a slow one (20 seconds).

### What we want instead

```
User: "Search for chicken products AND quick recipes"

  Main Agent
    |-- dispatches: search_products("chicken")  -> {status: dispatched}
    |-- dispatches: search_recipes("chicken")   -> {status: dispatched}
    |-- responds: "I'm searching for you..."    -> streamed to user immediately
    |
    |   ... 1 second later ...
    |
    |-- receives: product results               -> streamed to user
    |-- responds: "Here are some products: ..."
    |
    |   ... 19 seconds later ...
    |
    |-- receives: recipe results                -> streamed to user
    '-- responds: "And here are recipes: ..."
```

Each sub-agent's results are streamed to the user **as soon as they're available**,
independently of other pending tasks.

## The Solution: Follow-up FunctionResponses

The key insight: **a single FunctionCall can receive multiple FunctionResponses
over time**. This is how the LLM conversation protocol handles it:

```
Turn 1 (initial):
  user:   "Search for chicken"
  model:  FC: search_products(query="chicken")     <-- FunctionCall
  user:   FR: {status: "dispatched", task_id: "x"} <-- Immediate FunctionResponse
  model:  "I'm searching..."

Turn 2 (follow-up, when results arrive):
  user:   "Search for chicken"
  model:  FC: search_products(query="chicken")      <-- Same FunctionCall
  user:   FR: {status: "completed", results: [...]} <-- NEW FunctionResponse (replaces old)
  model:  "Here are your products: ..."
```

The framework replaces the original FunctionResponse with the new one and
re-runs the agent. The LLM sees the same FunctionCall but with updated results.

### Architecture

```
                        +-------------------------+
  SSE Stream to Client  |  AsyncOrchestratorGen   |
<-----------------------+                         |
                        |  1. Runs main agent     |
  Event: "Searching..." |  2. Detects dispatches  |
<-----------------------+  3. Starts sub-agents   |
                        |     as background tasks |
  Event: products       |  4. On completion:      |
<-----------------------+     sends follow-up FR  |
                        |  5. Yields all events   |
  Event: recipes        |  6. Completes when all  |
<-----------------------+     tasks are done      |
                        +-----------+-------------+
                                    |
                     +--------------+--------------+
                     v              v               v
               Main Agent    Product Agent    Recipe Agent
              (orchestrates)  (real LLM)       (real LLM)
```

The `run_with_async_tools()` async generator wraps the framework's runner and
keeps yielding events until all dispatched background tasks have completed.
This maps directly onto an SSE endpoint -- the stream stays open until everything
is done.

### Why this works

- **No synthetic/fake tool calls** -- follow-up responses go to the *original*
  dispatch calls that the LLM itself made
- **No extra tools to guard** -- no `receive_message` tool that the LLM might
  try to call on its own
- **Natural correlation** -- results are linked to their dispatch via `call_id`
- **No context pollution** -- no failed tool calls cluttering the conversation
- **Framework-native** -- uses the existing FunctionCall/FunctionResponse protocol

## Open Questions

- **Multi-model compatibility**: Does this pattern work across different LLMs?
  The ability to send multiple FunctionResponses for the same FunctionCall ID
  may be framework/model-specific. Validated so far:
  - Google Gemini 2.0 Flash (via ADK) -- works
  - Google Gemini 2.5 Flash (via ADK) -- works
  - OpenAI GPT-4o (via Agents SDK) -- untested
  - Anthropic Claude (via MCP/tool_use) -- untested
- **Conversation continuation**: Can the user send follow-up messages *while*
  background tasks are still running? This requires careful concurrency handling.
- **Error handling**: What happens when a sub-agent fails? The main agent needs
  to receive an error FunctionResponse and communicate it to the user.

## Examples

### Google ADK

See [`examples/google_adk/`](./examples/google_adk/) -- a complete working
example with real ADK sub-agents running as background tasks and a single async
generator that yields all events.

```bash
# Prerequisites: GCP credentials with Vertex AI access
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=europe-west1

uv run python examples/google_adk/main.py
```

## Setup

```bash
uv sync
```
