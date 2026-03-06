"""Proof: Gemini 2.5 Flash accepts follow-up FunctionResponses on the same call ID.

This test demonstrates the core async tool call pattern:

  1. Model makes a FunctionCall
  2. We respond immediately with {status: "dispatched"}
  3. Model acknowledges and responds to the user
  4. Later, we send a NEW FunctionResponse with the same call_id and real results
  5. Model processes the follow-up results and presents them

If step 4-5 succeed without API errors, the LLM's function calling protocol
supports non-blocking tool execution.

SDK: google-genai (https://github.com/googleapis/python-genai)
"""

from google.genai import types

# Use a single tool to keep the proof minimal and deterministic.
TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_products",
                description="Search for grocery products matching a query.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={"query": types.Schema(type="STRING")},
                    required=["query"],
                ),
            ),
        ]
    )
]

CONFIG = types.GenerateContentConfig(
    tools=TOOLS,
    system_instruction=(
        "You are a shopping assistant. When asked about products, "
        "call search_products. Present results clearly when they arrive."
    ),
)


def _has_function_call(response) -> bool:
    """Check if a response contains function calls."""
    return bool(response.function_calls)


def _has_text(response) -> bool:
    """Check if a response contains text."""
    return bool(response.text)


def _get_text(response) -> str:
    """Extract all text parts from a response."""
    texts = []
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.text:
                texts.append(part.text)
    return " ".join(texts)


def _handle_function_calls(chat, response) -> "GenerateContentResponse":
    """If the model makes function calls, send dummy responses and return next response."""
    while _has_function_call(response):
        parts = [
            types.Part(
                function_response=types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response={"result": "ok"},
                )
            )
            for fc in response.function_calls
        ]
        response = chat.send_message(parts)
    return response


def test_followup_function_response(client):
    """A follow-up FunctionResponse on the same call_id is accepted and processed."""
    chat = client.chats.create(model="gemini-2.5-flash", config=CONFIG)

    # ── Step 1: Trigger a function call ────────────────────────────────── #
    response = chat.send_message("Find chicken products")
    fcs = response.function_calls
    assert fcs and len(fcs) >= 1, "Model made no function calls"
    original_fc = fcs[0]
    original_call_id = original_fc.id

    # ── Step 2: Send immediate "dispatched" response ─────────────────── #
    response = chat.send_message(
        types.Part(
            function_response=types.FunctionResponse(
                id=original_fc.id,
                name=original_fc.name,
                response={"status": "dispatched", "message": "Search started"},
            )
        )
    )
    # Model may respond with text OR more function calls — both are fine.
    # If it makes more function calls, handle them to get back to a text state.
    response = _handle_function_calls(chat, response)
    assert _has_text(response), "Model did not produce any text after dispatched status"

    # ── Step 3: Send follow-up FunctionResponse with real results ─────── #
    # THIS IS THE CORE PROOF: we send a new FunctionResponse referencing
    # the ORIGINAL call_id from step 1. If the API accepts this without
    # error and the model processes the results, the pattern works.
    followup_results = {
        "status": "completed",
        "results": [
            {"name": "Chicken Breast 500g", "price": 4.99},
            {"name": "Chicken Thighs 600g", "price": 3.49},
        ],
    }
    response = chat.send_message(
        types.Part(
            function_response=types.FunctionResponse(
                id=original_call_id,
                name="search_products",
                response=followup_results,
            )
        )
    )

    # The model accepted the follow-up FR. It may respond with text
    # presenting the results, or with more function calls.
    response = _handle_function_calls(chat, response)
    text = _get_text(response).lower()
    assert text, "Model produced no text after follow-up FunctionResponse"

    # Verify the model actually used the data from the follow-up
    assert any(
        kw in text for kw in ["chicken", "4.99", "3.49", "breast", "thigh"]
    ), f"Model did not reference the follow-up results: {text}"
