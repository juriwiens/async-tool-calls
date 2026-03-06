# Async Tool Calls

Experiments with **asynchronous, non-blocking tool calls** for LLMs.

## The Problem

LLM function calling is inherently **synchronous and blocking**. When a model
makes a tool call, it expects the FunctionResponse in the immediately next
conversation turn — and it cannot proceed until all parallel tool calls have
been answered. This is a constraint of the **LLM conversation protocol itself**,
not just the frameworks built on top of it.

In practice, this means:

```
User: "Search for chicken products AND quick recipes"

  LLM
    |-- calls: search_products("chicken")     --+
    |-- calls: search_recipes("chicken")       -+  Both run in parallel,
    |                                           |  BUT the LLM only sees
    |         ... waiting for BOTH ...          |  results when ALL complete
    |                                           |
    |-- receives: products (after 1s)    -------+  Has to wait
    |-- receives: recipes  (after 20s)   -------+  for the slowest
    |
    '-- finally responds to user (after 20s)
```

**The problem**: If two tool calls run in parallel, the LLM can only process
and stream results to the user **after the slowest one finishes**. A fast tool
(1 second) is held hostage by a slow one (20 seconds).

This matters for any slow tool — external API calls, database queries, web
scraping, human-in-the-loop approvals, or sub-agent invocations.

### What we want instead

```
User: "Search for chicken products AND quick recipes"

  LLM
    |-- calls: search_products("chicken")   -> {status: dispatched}
    |-- calls: search_recipes("chicken")    -> {status: dispatched}
    |-- responds: "I'm searching for you..."  -> streamed to user immediately
    |
    |   ... 1 second later ...
    |
    |-- receives: product results             -> streamed to user
    |-- responds: "Here are some products: ..."
    |
    |   ... 19 seconds later ...
    |
    |-- receives: recipe results              -> streamed to user
    '-- responds: "And here are recipes: ..."
```

Each tool's results are delivered to the LLM and streamed to the user **as soon
as they're available**, independently of other pending tool calls.

## The Solution: Follow-up FunctionResponses

The key insight: **a single FunctionCall can receive multiple FunctionResponses
over time**. The tool returns an immediate "dispatched" response, and the actual
result is delivered later as a follow-up response to the same call ID.

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

The application layer replaces the original FunctionResponse with the new one
and re-invokes the LLM. The model sees the same FunctionCall but with updated
results.

### Architecture (example with sub-agents as tools)

The Google ADK example demonstrates this with sub-agents, but the pattern
applies to any slow tool:

```
                        +-------------------------+
  SSE Stream to Client  |  AsyncOrchestratorGen   |
<-----------------------+                         |
                        |  1. Runs LLM            |
  Event: "Searching..." |  2. Detects dispatched  |
<-----------------------+     tool calls          |
                        |  3. Starts async work   |
  Event: products       |  4. On completion:      |
<-----------------------+     sends follow-up FR  |
                        |  5. Yields all events   |
  Event: recipes        |  6. Completes when all  |
<-----------------------+     tasks are done      |
                        +-----------+-------------+
                                    |
                     +--------------+--------------+
                     v              v               v
                  LLM Agent    Async Work 1    Async Work 2
                               (API call,      (sub-agent,
                                DB query,       web scrape,
                                sub-agent...)   approval...)
```

The `run_with_async_tools()` async generator wraps the framework's runner and
keeps yielding events until all dispatched async work has completed. This maps
directly onto an SSE endpoint — the stream stays open until everything is done.

### Why this works

- **No synthetic/fake tool calls** — follow-up responses go to the *original*
  tool calls that the LLM itself made
- **No extra tools to guard** — no hidden `receive_message` tool that the LLM
  might try to call on its own
- **Natural correlation** — results are linked to their dispatch via `call_id`
- **No context pollution** — no failed tool calls cluttering the conversation
- **Protocol-native** — uses the existing FunctionCall/FunctionResponse protocol

## Open Questions

- **Multi-model compatibility**: Does this pattern work across different LLMs?
  The ability to send multiple FunctionResponses for the same FunctionCall ID
  may be model/framework-specific. Validated so far:
  - Google Gemini 2.0 Flash (via ADK) -- works
  - Google Gemini 2.5 Flash (via ADK) -- works
  - OpenAI GPT-4o (via Agents SDK) -- untested
  - Anthropic Claude (via MCP/tool_use) -- untested
- **Conversation continuation**: Can the user send follow-up messages *while*
  async tools are still running? This requires careful concurrency handling.
- **Error handling**: What happens when an async tool fails? The LLM needs
  to receive an error FunctionResponse and communicate it to the user.

## Examples

### Google ADK

See [`examples/google_adk/`](./examples/google_adk/) — a working example
that uses real ADK sub-agents as the async work behind dispatched tool calls.

```bash
# Prerequisites: GCP credentials with Vertex AI access
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=europe-west1

uv run python -m examples.google_adk.main
```

## Setup

```bash
uv sync
```
