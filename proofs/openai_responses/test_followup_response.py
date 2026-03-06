"""Proof: GPT-5.4 via OpenAI Responses API — follow-up function_call_output.

Tests whether the Responses API accepts a function_call_output in a later turn
that references a call_id from an earlier function_call (not the immediately
preceding one).

SDK: openai (https://github.com/openai/openai-python)
API: Responses API (via Codex endpoint or Platform API)
Model: gpt-5.4
"""

import json

import pytest
from openai import BadRequestError

MODEL = "gpt-5.4"

INSTRUCTIONS = (
    "You are a shopping assistant. When asked about products, "
    "call search_products. Present results clearly when they arrive."
)

TOOLS = [
    {
        "type": "function",
        "name": "search_products",
        "description": "Search for grocery products matching a query.",
        "parameters": {
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


def _create_response(client, input_list):
    """Call responses.stream (returns ResponseStreamManager with get_final_response support)."""
    with client.responses.stream(
        model=MODEL,
        tools=TOOLS,
        instructions=INSTRUCTIONS,
        input=input_list,
        store=False,
    ) as stream:
        response = stream.get_final_response()

    return response


def _extract_text(response) -> str:
    texts = []
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if hasattr(part, "text"):
                    texts.append(part.text)
    return " ".join(texts)


def _extract_function_calls(response) -> list:
    return [item for item in response.output if item.type == "function_call"]


def test_followup_function_call_output(client):
    """A follow-up function_call_output on the same call_id — does the API accept it?"""
    input_list = [
        {"role": "user", "content": "Find chicken products"},
    ]

    # ── Step 1: Trigger a function call ────────────────────────────────── #
    response = _create_response(client, input_list)
    fcs = _extract_function_calls(response)
    assert fcs, "Model did not make any function calls"
    original_fc = fcs[0]
    original_call_id = original_fc.call_id

    # ── Step 2: Send immediate "dispatched" output ─────────────────────── #
    input_list += response.output
    input_list.append({
        "type": "function_call_output",
        "call_id": original_call_id,
        "output": json.dumps({"status": "dispatched", "message": "Search started"}),
    })

    response = _create_response(client, input_list)

    # Handle any extra function calls to get to a text response
    while _extract_function_calls(response):
        input_list += response.output
        for fc in _extract_function_calls(response):
            input_list.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": json.dumps({"result": "ok"}),
            })
        response = _create_response(client, input_list)

    intermediate_text = _extract_text(response)
    assert intermediate_text, "Model did not produce text after dispatched status"
    input_list += response.output

    # ── Step 3: Send follow-up function_call_output with real results ── #
    input_list.append({
        "type": "function_call_output",
        "call_id": original_call_id,
        "output": json.dumps(PRODUCT_RESULTS),
    })

    try:
        response = _create_response(client, input_list)
    except BadRequestError as e:
        pytest.fail(
            f"API rejected follow-up function_call_output: {e.message}"
        )

    # Handle any extra function calls
    while _extract_function_calls(response):
        input_list += response.output
        for fc in _extract_function_calls(response):
            input_list.append({
                "type": "function_call_output",
                "call_id": fc.call_id,
                "output": json.dumps({"result": "ok"}),
            })
        response = _create_response(client, input_list)

    followup_text = _extract_text(response)
    assert followup_text, "Model produced no text after follow-up function_call_output"

    text_lower = followup_text.lower()
    assert any(
        kw in text_lower for kw in ["chicken", "4.99", "3.49", "breast", "thigh"]
    ), f"Model did not reference the follow-up results: {followup_text}"
