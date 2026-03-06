"""Proof: Claude on Vertex AI — follow-up tool_result behavior.

FINDING: The Anthropic Messages API REJECTS follow-up tool_results that
reference a tool_use_id from a non-immediately-preceding assistant message.

    "Each tool_result block must have a corresponding tool_use block
     in the previous message."

This means the Gemini-style follow-up pattern (send a new FunctionResponse
for the same call_id in a later turn) does NOT work with Claude.

ALTERNATIVE: History replacement — replace the original tool_result in the
conversation history with the completed result and re-invoke the model.
This approach works but requires the application layer to manage history.

SDK: anthropic[vertex] (https://github.com/anthropics/anthropic-sdk-python)
Models: claude-haiku-4-5@20251001, claude-sonnet-4-6
"""

import copy
import json

import anthropic
import pytest

MODELS = [
    "claude-haiku-4-5@20251001",
    "claude-sonnet-4-6",
]

TOOLS = [
    {
        "name": "search_products",
        "description": "Search for grocery products matching a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
]

PRODUCT_RESULTS = {
    "status": "completed",
    "results": [
        {"name": "Chicken Breast 500g", "price": 4.99},
        {"name": "Chicken Thighs 600g", "price": 3.49},
    ],
}


def _extract_text(response) -> str:
    return " ".join(
        block.text for block in response.content if block.type == "text"
    )


def _extract_tool_use(response):
    for block in response.content:
        if block.type == "tool_use":
            return block
    return None


@pytest.mark.parametrize("model", MODELS)
def test_followup_tool_result_is_rejected(client, model):
    """Claude API rejects a tool_result that references a non-preceding tool_use.

    Conversation:
      1. user:      "Find chicken products"
      2. assistant:  [tool_use id=X]
      3. user:       [tool_result tool_use_id=X -> dispatched]
      4. assistant:  "I'm searching..."  (text only)
      5. user:       [tool_result tool_use_id=X -> completed]  ← REJECTED
    """
    messages = []

    # Step 1: Trigger tool call
    messages.append({"role": "user", "content": "Find chicken products"})
    response = client.messages.create(
        model=model, max_tokens=1024, tools=TOOLS, messages=messages,
    )
    tool_use = _extract_tool_use(response)
    assert tool_use is not None, "Model did not make a tool call"
    original_tool_use_id = tool_use.id

    # Step 2: Send "dispatched" tool_result
    messages.append({"role": "assistant", "content": response.content})
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": original_tool_use_id,
            "content": json.dumps({"status": "dispatched"}),
        }],
    })
    response = client.messages.create(
        model=model, max_tokens=1024, tools=TOOLS, messages=messages,
    )

    # Handle any extra tool calls to get to a text response
    while response.stop_reason == "tool_use":
        tu = _extract_tool_use(response)
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tu.id, "content": "ok"}],
        })
        response = client.messages.create(
            model=model, max_tokens=1024, tools=TOOLS, messages=messages,
        )

    messages.append({"role": "assistant", "content": response.content})

    # Step 3: Attempt follow-up tool_result → expect rejection
    messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": original_tool_use_id,
            "content": json.dumps(PRODUCT_RESULTS),
        }],
    })

    with pytest.raises(anthropic.BadRequestError, match="tool_use_id"):
        client.messages.create(
            model=model, max_tokens=1024, tools=TOOLS, messages=messages,
        )


@pytest.mark.parametrize("model", MODELS)
def test_history_replacement_works(client, model):
    """Alternative: replace the dispatched tool_result in history and re-invoke.

    Instead of sending a follow-up tool_result, we rewrite the conversation
    history to replace the original "dispatched" response with the real
    results, then re-invoke the model. The model sees the same tool_use
    but with updated results.

    Conversation (as seen by the model on re-invocation):
      1. user:      "Find chicken products"
      2. assistant:  [tool_use id=X]
      3. user:       [tool_result tool_use_id=X -> completed, results=[...]]
      ↑ REPLACED (was "dispatched", now "completed")
    """
    messages = []

    # Step 1: Trigger tool call
    messages.append({"role": "user", "content": "Find chicken products"})
    response = client.messages.create(
        model=model, max_tokens=1024, tools=TOOLS, messages=messages,
    )
    tool_use = _extract_tool_use(response)
    assert tool_use is not None, "Model did not make a tool call"

    # Step 2: Send "dispatched" tool_result (initial fast response)
    messages.append({"role": "assistant", "content": response.content})
    tool_result_turn = {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": json.dumps({"status": "dispatched"}),
        }],
    }
    messages.append(tool_result_turn)
    response = client.messages.create(
        model=model, max_tokens=1024, tools=TOOLS, messages=messages,
    )
    dispatched_text = _extract_text(response)
    assert dispatched_text, "Model did not respond after dispatched status"

    # Step 3: Replace the tool_result in history and re-invoke
    # Deep copy to avoid mutating the original, then truncate to
    # just [user_msg, assistant_tool_use, user_tool_result]
    replaced_messages = copy.deepcopy(messages[:3])
    replaced_messages[2] = {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": json.dumps(PRODUCT_RESULTS),
        }],
    }

    response = client.messages.create(
        model=model, max_tokens=1024, tools=TOOLS, messages=replaced_messages,
    )

    # Handle any extra tool calls
    while response.stop_reason == "tool_use":
        tu = _extract_tool_use(response)
        replaced_messages.append({"role": "assistant", "content": response.content})
        replaced_messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tu.id, "content": "ok"}],
        })
        response = client.messages.create(
            model=model, max_tokens=1024, tools=TOOLS, messages=replaced_messages,
        )

    followup_text = _extract_text(response)
    assert followup_text, "Model produced no text after history replacement"

    text_lower = followup_text.lower()
    assert any(
        kw in text_lower for kw in ["chicken", "4.99", "3.49", "breast", "thigh"]
    ), f"Model did not reference the results: {followup_text}"
