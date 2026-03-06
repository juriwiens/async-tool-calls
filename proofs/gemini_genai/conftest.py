"""Shared fixtures for Gemini python-genai proofs."""

import os

import pytest
from google import genai


@pytest.fixture
def client():
    """Create a google-genai client.

    Supports two auth modes via environment variables:
      - GOOGLE_API_KEY          → API key auth (simplest)
      - GOOGLE_GENAI_USE_VERTEXAI=1 → Vertex AI auth (requires
        GOOGLE_CLOUD_PROJECT and optionally GOOGLE_CLOUD_LOCATION)
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"):
        return genai.Client(
            vertexai=True,
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )

    pytest.skip("Set GOOGLE_API_KEY or GOOGLE_GENAI_USE_VERTEXAI=1")
