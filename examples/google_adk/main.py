"""
Run the async tool calls example with Google ADK.

Simulates an SSE stream by consuming the async orchestrator generator
and printing events as they arrive.
"""

import asyncio
import os

from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

from .agents import main_agent
from .orchestrator import run_with_async_tools


async def main():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=main_agent,
        app_name="async_tool_calls_demo",
        session_service=session_service,
    )
    user_id = "demo-user"
    session = await session_service.create_session(
        app_name="async_tool_calls_demo", user_id=user_id,
    )

    # ── First request: triggers async dispatches ────────────────────────── #
    print("=" * 70)
    print("SSE STREAM 1: Initial request")
    print("=" * 70)

    event_count = 0
    async for event in run_with_async_tools(
        runner, session, user_id,
        message="Ich moechte heute Abend Haehnchen kochen. "
                "Suche mir passende Produkte und ein schnelles Rezept.",
    ):
        event_count += 1
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"\n  [{event.author}] {part.text}")
                if part.function_call:
                    print(f"  [{event.author}] FC: {part.function_call.name}"
                          f"({dict(part.function_call.args or {})})")
                if part.function_response:
                    resp = part.function_response.response
                    if isinstance(resp, dict):
                        status = resp.get("status", "?")
                        if status == "dispatched":
                            print(f"  [{event.author}] FR: "
                                  f"{part.function_response.name} -> dispatched")
                        elif status == "completed":
                            n = len(resp.get("results", []))
                            print(f"  [{event.author}] FR: "
                                  f"{part.function_response.name} -> "
                                  f"completed ({n} results)")

    print(f"\nStream 1 complete. {event_count} events.\n")

    # ── Follow-up request: cross-references previous results ────────────── #
    print("=" * 70)
    print("SSE STREAM 2: Follow-up request")
    print("=" * 70)

    async for event in run_with_async_tools(
        runner, session, user_id,
        message="Welche Produkte brauche ich fuer das erste Rezept "
                "und was wuerde das kosten?",
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"\n  [{event.author}] {part.text}")

    print()


if __name__ == "__main__":
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "rd-stationary-services-int")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "europe-west1")
    asyncio.run(main())
