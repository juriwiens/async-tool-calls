"""Debug script: shows the full conversation at each step."""

import os
import json
from google import genai
from google.genai import types

# ── Client setup ──────────────────────────────────────────────────────── #
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )

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


def dump_response(label: str, response):
    """Pretty-print all parts of a response."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    for ci, candidate in enumerate(response.candidates or []):
        for pi, part in enumerate(candidate.content.parts or []):
            if part.text:
                print(f"  [part {pi}] TEXT: {part.text}")
            if part.function_call:
                fc = part.function_call
                print(f"  [part {pi}] FUNCTION_CALL:")
                print(f"    name: {fc.name}")
                print(f"    id:   {fc.id}")
                print(f"    args: {dict(fc.args or {})}")
            if part.function_response:
                fr = part.function_response
                print(f"  [part {pi}] FUNCTION_RESPONSE:")
                print(f"    name: {fr.name}")
                print(f"    id:   {fr.id}")
    print()


def dump_history(chat):
    """Print the full chat history."""
    print(f"\n{'═' * 60}")
    print("  FULL CHAT HISTORY")
    print(f"{'═' * 60}")
    for ti, content in enumerate(chat._curated_history):
        print(f"\n  Turn {ti} (role={content.role}):")
        for pi, part in enumerate(content.parts or []):
            if part.text:
                print(f"    [part {pi}] TEXT: {part.text[:120]}...")
            if part.function_call:
                fc = part.function_call
                print(f"    [part {pi}] FC: {fc.name}(id={fc.id}, args={dict(fc.args or {})})")
            if part.function_response:
                fr = part.function_response
                resp_str = json.dumps(fr.response, default=str)[:120]
                print(f"    [part {pi}] FR: {fr.name}(id={fr.id}) -> {resp_str}")
    print()


# ── Run the conversation ──────────────────────────────────────────────── #
chat = client.chats.create(model="gemini-2.5-flash", config=CONFIG)

# Step 1: Trigger function call
print("\n\n>>> STEP 1: Sending user message to trigger function call")
response = chat.send_message("Find chicken products")
dump_response("STEP 1 RESPONSE (expect: function_call)", response)

fcs = response.function_calls
assert fcs, "Model made no function calls!"
original_fc = fcs[0]
print(f"  → Got function call: {original_fc.name}(id={original_fc.id})")

# Step 2: Send "dispatched" response
print("\n\n>>> STEP 2: Sending 'dispatched' FunctionResponse")
response = chat.send_message(
    types.Part(
        function_response=types.FunctionResponse(
            id=original_fc.id,
            name=original_fc.name,
            response={"status": "dispatched", "message": "Search started"},
        )
    )
)
dump_response("STEP 2 RESPONSE (expect: text acknowledgment)", response)

# Step 3: Send follow-up FunctionResponse with real results
print("\n\n>>> STEP 3: Sending FOLLOW-UP FunctionResponse (same call_id!)")
print(f"  → Using original call_id: {original_fc.id}")
response = chat.send_message(
    types.Part(
        function_response=types.FunctionResponse(
            id=original_fc.id,
            name="search_products",
            response={
                "status": "completed",
                "results": [
                    {"name": "Chicken Breast 500g", "price": 4.99},
                    {"name": "Chicken Thighs 600g", "price": 3.49},
                ],
            },
        )
    )
)
dump_response("STEP 3 RESPONSE (the proof — does it work?)", response)

# If model made another function call, show that too
if response.function_calls:
    print("  ⚠ Model made ANOTHER function call instead of presenting results!")
    print("  → Sending dummy response to get text...")
    response = chat.send_message([
        types.Part(
            function_response=types.FunctionResponse(
                id=fc.id,
                name=fc.name,
                response={"result": "no more results"},
            )
        )
        for fc in response.function_calls
    ])
    dump_response("STEP 3b RESPONSE (after handling extra FC)", response)

# Show full history
dump_history(chat)
