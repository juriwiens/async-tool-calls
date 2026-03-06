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

Think of JavaScript's **event loop**: the main thread never blocks on I/O. It
dispatches work, stays responsive, and processes results as they arrive. We want
the same for LLM agents — the model should **never be blocked** by a pending
tool call. It dispatches work, immediately responds to the user, and remains
free to process each result as it completes:

```
User: "Search for chicken products AND quick recipes"

  LLM (always free, never blocking)
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

Just like a JavaScript event loop, the agent is **always ready to respond** —
it never sits idle waiting for a slow operation. Each tool's results are
delivered to the LLM and streamed to the user **as soon as they're available**,
independently of other pending tool calls.

## Solution Approaches

All approaches share the same first step: the tool returns an immediate
"dispatched" response so the model can respond to the user right away.
They differ in **how the real results are delivered later**.

### Approach A: Follow-up FunctionResponses

Send a **new FunctionResponse** for the same call ID in a later conversation
turn. The model receives the updated results as if the tool had just completed.

```
Turn 1 (dispatch):
  user:   "Search for chicken"
  model:  FC: search_products(query="chicken")       id=X
  user:   FR: {status: "dispatched"}                  id=X
  model:  "I'm searching..."

Turn 2 (follow-up, when results arrive):
  user:   FR: {status: "completed", results: [...]}   id=X  ← new FR, same id
  model:  "Here are your products: ..."
```

| | |
|---|---|
| ✅ Pros | No history rewriting, no synthetic tool calls, clean conversation flow |
| ❌ Cons | **Not universally supported** — requires the API to accept a FunctionResponse without a matching FunctionCall in the immediately preceding turn |
| 📊 Support | Gemini ✅ · Claude ❌ · OpenAI untested |

### Approach B: History Replacement

When results arrive, **replace** the original "dispatched" FunctionResponse in
the conversation history with the real results, then re-invoke the model. The
model sees the same FunctionCall but with updated data — as if the tool had
always returned the complete result.

```
History before replacement:
  user:   "Search for chicken"
  model:  FC: search_products(query="chicken")
  user:   FR: {status: "dispatched"}              ← will be replaced
  model:  "I'm searching..."

History after replacement:
  user:   "Search for chicken"
  model:  FC: search_products(query="chicken")
  user:   FR: {status: "completed", results: [...]}  ← replaced
  (model turn removed — will be re-generated)
```

| | |
|---|---|
| ✅ Pros | Works with **any** LLM API — only requires standard tool calling support |
| ❌ Cons | Requires application-layer history management; model re-generates its response (no incremental update); intermediate "I'm searching..." response is lost from history |
| 📊 Support | Gemini ✅ · Claude ✅ · OpenAI untested |

> **⚠️ Critical limitation**: History Replacement only works if **nothing else
> happened** between dispatch and completion. If the user sent follow-up
> messages, other tools completed, or the model produced intermediate responses,
> you face a dilemma:
>
> - **Truncate** the history back to the tool result → lose all intermediate
>   interactions
> - **Replace in-place** without truncating → the history becomes inconsistent
>   (the tool result says "completed" but the model's next response still says
>   "I'm searching...")
>
> This makes History Replacement viable only as an **immediate-response
> strategy** (dispatch → replace → re-invoke with no gap), not for true
> asynchronous behavior where the conversation continues while tools run.

### Approach C: Injected Tool Calls

Inject a **synthetic** FunctionCall + FunctionResponse pair into the
conversation history. The application fabricates a model turn containing a
FunctionCall (e.g. `receive_results`) that the model never actually made,
paired with a user turn containing the FunctionResponse with the real results.

```
History after injection:
  user:   "Search for chicken"
  model:  FC: search_products(query="chicken")
  user:   FR: {status: "dispatched"}
  model:  "I'm searching..."
  model:  FC: receive_results(task_id="prod-abc")     ← injected (model never made this)
  user:   FR: {results: [...]}                         ← injected
```

| | |
|---|---|
| ✅ Pros | Works with any LLM API; preserves the full conversation history including intermediate responses |
| ❌ Cons | **Context pollution** — the model sees tool calls it never made, which can cause it to start calling `receive_results` on its own; requires guarding (e.g. `before_tool_callback`) or keeping the tool unregistered, both with trade-offs; fabricated model turns may confuse the model's self-model |
| 📊 Support | Technically works everywhere, but fragile and not recommended |

### Approach D: Structured User Messages

Bypass the tool calling protocol entirely. Deliver results as a **regular user
message** with a structured format (e.g. XML) that the model is instructed to
recognize via system instructions.

The key challenge: the model must understand that this message **is not from the
user** and the **user has not seen its contents**. It's effectively a system
notification injected into the user turn. Without this distinction, the model
might respond with "Thanks for the data!" instead of presenting the results.

```
System instruction:
  "Messages wrapped in <async_tool_result> are system notifications,
   NOT user messages. The user has not seen this data. When you receive
   one, present the results to the user in a helpful way.
   Never ask the user about the raw data — they don't see it."

History:
  user:   "Search for chicken"
  model:  FC: search_products(query="chicken")
  user:   FR: {status: "dispatched"}
  model:  "I'm searching..."
  user:   "<async_tool_result tool="search_products"            ← NOT from the user
             task_id="prod-abc">
            [{"name": "Chicken Breast", "price": 4.99}, ...]
          </async_tool_result>"
  model:  "Here are your products: ..."
```

| | |
|---|---|
| ✅ Pros | Works with **any** LLM API; no history manipulation; survives intermediate interactions; append-only (no rewrites); no synthetic tool calls |
| ❌ Cons | **Relies on prompt engineering** — the model must be instructed to treat these messages as system notifications, not user input; results are not protocol-native; no structured correlation via `call_id`; risk of the model leaking raw data or addressing the "sender" |
| 📊 Support | Works everywhere — it's just a user message |

### Comparison

| | Follow-up FR | History Replacement | Injected Tool Calls | Structured User Msg |
|---|:---:|:---:|:---:|:---:|
| Universal API support | ❌ | ✅ | ✅ | ✅ |
| Clean history | ✅ | ⚠️ loses intermediate | ❌ pollutes context | ✅ |
| No synthetic turns | ✅ | ✅ | ❌ | ✅ |
| Model can't misuse | ✅ | ✅ | ❌ needs guarding | ✅ |
| Incremental updates | ✅ | ❌ re-generates | ✅ | ✅ |
| Survives intermediate interactions | ✅ | ❌ | ✅ | ✅ |
| Protocol-native | ✅ | ✅ | ⚠️ fabricated | ❌ prompt-based |
| No prompt engineering needed | ✅ | ✅ | ✅ | ❌ |

**Recommendation**: Use **Follow-up FR** (Approach A) where the API supports it
(Gemini). For APIs that don't (Claude), the choice depends on the use case:
- **No intermediate interactions** (dispatch → wait → deliver): use **History
  Replacement** (Approach B) — simple and clean.
- **Conversation continues while tools run**: choose between **Injected Tool
  Calls** (Approach C) and **Structured User Messages** (Approach D).
  Approach D is simpler and cleaner but relies on prompt engineering;
  Approach C uses the native protocol but pollutes context.

### Architecture

Regardless of which approach is used to deliver results, the orchestration
pattern is the same:

```
                        +-------------------------+
  SSE Stream to Client  |  AsyncOrchestratorGen   |
<-----------------------+                         |
                        |  1. Runs LLM            |
  Event: "Searching..." |  2. Detects dispatched  |
<-----------------------+     tool calls          |
                        |  3. Starts async work   |
  Event: products       |  4. On completion:      |
<-----------------------+     delivers results    |
                        |     (approach A, B or C) |
  Event: recipes        |  5. Yields all events   |
<-----------------------+  6. Completes when all  |
                        |     tasks are done      |
                        +-----------+-------------+
                                    |
                     +--------------+--------------+
                     v              v               v
                  LLM Agent    Async Work 1    Async Work 2
                               (API call,      (sub-agent,
                                DB query,       web scrape,
                                sub-agent...)   approval...)
```

The orchestrator wraps the LLM runner as an **async generator** and keeps
yielding events until all dispatched work has completed. This maps directly onto
an SSE endpoint — the stream stays open until everything is done.

## Open Questions

- **Conversation continuation**: Can the user send follow-up messages *while*
  async tools are still running? This requires careful concurrency handling.
- **Error handling**: What happens when an async tool fails? The LLM needs
  to receive an error FunctionResponse and communicate it to the user.

## Proofs

Each proof validates the follow-up FunctionResponse pattern against a specific
model using its native SDK. Run them with pytest:

```bash
uv sync
uv run pytest proofs/ -v
```

| Model | SDK | Follow-up FR | History Replacement | Proof |
|-------|-----|:------------:|:-------------------:|-------|
| Gemini 2.5 Flash | [python-genai](https://github.com/googleapis/python-genai) | ✅ | — | [`proofs/gemini_genai/`](./proofs/gemini_genai/) |
| Claude Haiku 4.5 | [anthropic\[vertex\]](https://github.com/anthropics/anthropic-sdk-python) | ❌ | ✅ | [`proofs/claude_vertex/`](./proofs/claude_vertex/) |
| Claude Sonnet 4.6 | [anthropic\[vertex\]](https://github.com/anthropics/anthropic-sdk-python) | ❌ | ✅ | [`proofs/claude_vertex/`](./proofs/claude_vertex/) |
| OpenAI GPT-4o | openai | untested | untested | — |

**Follow-up FR** = send a new FunctionResponse/tool_result for the same call ID
in a later conversation turn (the approach described in this README).

**History Replacement** = rewrite the conversation history to replace the
original "dispatched" response with the real results, then re-invoke the model.

### Gemini 2.5 Flash (`python-genai`)

Follow-up FunctionResponses work natively. The model accepts a new FR on the
same call ID in a later turn without any API error.

```bash
# Option A: API key (simplest)
export GOOGLE_API_KEY=your-api-key

# Option B: Vertex AI
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1

uv run pytest proofs/gemini_genai/ -v
```

### Claude Haiku 4.5 (`anthropic[vertex]`)

The Anthropic Messages API **rejects** follow-up `tool_result` blocks that
reference a `tool_use_id` from a non-immediately-preceding assistant message:

> *"Each tool_result block must have a corresponding tool_use block
> in the previous message."*

**Workaround**: History replacement — replace the original "dispatched"
`tool_result` in the conversation history with the completed result and
re-invoke the model. The model sees the same `tool_use` but with updated data.

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=europe-west1

uv run pytest proofs/claude_vertex/ -v
```

## Examples

### Google ADK (Orchestrator)

See [`examples/google_adk/`](./examples/google_adk/) — a full orchestrator
example that uses real ADK sub-agents as async work behind dispatched tool calls,
with an async generator that maps directly onto an SSE endpoint.

## Setup

```bash
uv sync
```
