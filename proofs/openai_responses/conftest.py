"""Shared fixtures for OpenAI Responses API proofs."""

import json
import os
import time
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
TOKEN_FILE = Path(__file__).parent.parent.parent / ".codex_tokens.json"


def _refresh_codex_tokens(refresh_token: str) -> dict:
    """Refresh expired Codex access token."""
    resp = httpx.post(
        f"{ISSUER}/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    return resp.json()


def _load_codex_tokens() -> dict | None:
    """Load and refresh Codex tokens if available."""
    if not TOKEN_FILE.exists():
        return None

    token_data = json.loads(TOKEN_FILE.read_text())

    # Refresh if expired (with 60s buffer)
    if token_data.get("expires_at", 0) < time.time() + 60:
        tokens = _refresh_codex_tokens(token_data["refresh_token"])
        token_data["access_token"] = tokens["access_token"]
        token_data["refresh_token"] = tokens["refresh_token"]
        token_data["expires_at"] = time.time() + tokens.get("expires_in", 3600)
        TOKEN_FILE.write_text(json.dumps(token_data, indent=2))

    return token_data


@pytest.fixture
def client():
    """Create an OpenAI client.

    Supports two auth modes:
      - OPENAI_API_KEY → standard OpenAI Platform API
      - .codex_tokens.json → ChatGPT subscription via Codex endpoint
        (run `uv run python -m proofs.openai_responses.login` first)
    """
    # Option 1: Standard API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)

    # Option 2: Codex tokens (ChatGPT subscription)
    tokens = _load_codex_tokens()
    if tokens:
        default_headers = {}
        if tokens.get("account_id"):
            default_headers["ChatGPT-Account-Id"] = tokens["account_id"]

        return OpenAI(
            api_key=tokens["access_token"],
            base_url=CODEX_BASE_URL,
            default_headers=default_headers,
        )

    pytest.skip(
        "Set OPENAI_API_KEY or run "
        "'uv run python -m proofs.openai_responses.login' first"
    )
