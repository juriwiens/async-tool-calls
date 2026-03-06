"""
Async orchestrator that wraps the ADK runner to support non-blocking tool calls.

The key primitive: run_with_async_tools() is an async generator that yields
events until ALL dispatched background tasks have completed. This maps directly
onto an SSE endpoint.
"""

import asyncio
import json
from typing import AsyncGenerator

from google.adk.agents.llm_agent import Agent
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from .agents import SUB_AGENT_MAP


async def _run_sub_agent(agent: Agent, query: str) -> dict:
    """Runs a sub-agent in isolation and returns its response as a dict."""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=f"sub_{agent.name}",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name=f"sub_{agent.name}", user_id="system",
    )

    result_text = ""
    async for event in runner.run_async(
        user_id="system",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=f"Search query: {query}")],
        ),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    result_text += part.text

    try:
        results = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        results = [{"raw_response": result_text}]

    return {"status": "completed", "results": results}


async def run_with_async_tools(
    runner: Runner,
    session,
    user_id: str,
    message: str,
) -> AsyncGenerator[Event, None]:
    """Async generator that yields events until ALL dispatched tasks complete.

    Usage (SSE endpoint):
        async for event in run_with_async_tools(runner, session, user_id, msg):
            yield f"data: {serialize(event)}\\n\\n"
        # Stream ends when all background tasks are done.

    How it works:
    1. Runs the main agent's initial turn (dispatches tool calls)
    2. Detects dispatched async tools from FunctionResponse events
    3. Starts real sub-agents as asyncio background tasks
    4. When a sub-agent completes, sends a follow-up FunctionResponse
       to the ORIGINAL tool call (same call_id)
    5. Yields all events from both the initial and follow-up runs
    6. Completes when no pending tasks remain
    """
    # Track pending async tasks: call_id -> asyncio.Task
    pending_tasks: dict[str, asyncio.Task] = {}
    # Queue for completed results
    results_queue: asyncio.Queue[tuple[str, str, dict]] = asyncio.Queue()

    # ── Phase 1: Initial run ────────────────────────────────────────────── #
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=message)],
        ),
    ):
        yield event

        # Detect dispatch FunctionResponses and start background sub-agents
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (part.function_response
                        and part.function_response.name in SUB_AGENT_MAP
                        and isinstance(part.function_response.response, dict)):
                    resp = part.function_response.response
                    if resp.get("status") == "dispatched":
                        call_id = part.function_response.id
                        tool_name = part.function_response.name
                        query = resp.get("query", "")
                        sub_agent = SUB_AGENT_MAP[tool_name]

                        async def _run_and_enqueue(cid, tn, q, agent):
                            result = await _run_sub_agent(agent, q)
                            await results_queue.put((cid, tn, result))

                        task = asyncio.create_task(
                            _run_and_enqueue(call_id, tool_name, query, sub_agent)
                        )
                        pending_tasks[call_id] = task

    if not pending_tasks:
        return

    # ── Phase 2: Yield follow-up events as sub-agents complete ──────────── #
    while pending_tasks:
        call_id, tool_name, result = await results_queue.get()
        pending_tasks.pop(call_id, None)

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id,
                        name=tool_name,
                        response=result,
                    )
                )],
            ),
        ):
            yield event
