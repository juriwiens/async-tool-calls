"""Agent definitions for the async tool calls example."""

import uuid

from google.adk.agents.llm_agent import Agent

# --------------------------------------------------------------------------- #
# Sub-Agents: independent ADK agents that do the actual work
# --------------------------------------------------------------------------- #

product_agent = Agent(
    name="product_agent",
    model="gemini-2.5-flash",
    instruction="""You are a product search agent for a German grocery store.
When given a search query, return a JSON list of 3-4 matching products.
Each product should have: name, price (in EUR), weight, and a short description.
Be creative but realistic for a German grocery store (e.g. REWE).
Respond ONLY with the JSON array, no other text.""",
)

recipe_agent = Agent(
    name="recipe_agent",
    model="gemini-2.5-flash",
    instruction="""You are a recipe search agent.
When given a search query, return a JSON list of 2-3 matching recipes.
Each recipe should have: name, time (preparation time), servings,
and ingredients (list of strings).
Be creative but realistic for German home cooking.
Respond ONLY with the JSON array, no other text.""",
)


# --------------------------------------------------------------------------- #
# Dispatch tools: return immediately, results arrive later via follow-up FR
# --------------------------------------------------------------------------- #

def dispatch_product_search(query: str) -> dict:
    """Searches for grocery products matching the query.
    Returns immediately with a task_id. Final results will arrive later
    as a follow-up response."""
    task_id = f"prod-{uuid.uuid4().hex[:8]}"
    return {"status": "dispatched", "task_id": task_id, "query": query}


def dispatch_recipe_search(query: str) -> dict:
    """Searches for recipes matching the query.
    Returns immediately with a task_id. Final results will arrive later
    as a follow-up response."""
    task_id = f"rec-{uuid.uuid4().hex[:8]}"
    return {"status": "dispatched", "task_id": task_id, "query": query}


# Map dispatch tool names to their sub-agents
SUB_AGENT_MAP = {
    "dispatch_product_search": product_agent,
    "dispatch_recipe_search": recipe_agent,
}


# --------------------------------------------------------------------------- #
# Main Agent: orchestrates by dispatching to sub-agents
# --------------------------------------------------------------------------- #

main_agent = Agent(
    name="main_agent",
    model="gemini-2.5-flash",
    instruction="""You are a helpful shopping assistant for a German grocery store.

When the user asks about products or cooking, you MUST use your tools:
- dispatch_product_search: to find grocery products
- dispatch_recipe_search: to find recipes

Always call the relevant tool(s) first. Both return immediately with a
pending status. The actual results will arrive later as follow-up responses.

When results arrive, present them to the user in a friendly, helpful way.
When you have both products and recipes, help the user connect them
(e.g. which products are needed for which recipe).

Always respond in German.""",
    tools=[dispatch_product_search, dispatch_recipe_search],
)
