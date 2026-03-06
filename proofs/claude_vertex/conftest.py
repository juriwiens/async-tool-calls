"""Shared fixtures for Claude on Vertex AI proofs."""

import os

import pytest
from anthropic import AnthropicVertex


@pytest.fixture
def client():
    """Create an AnthropicVertex client.

    Requires GCP Application Default Credentials and:
      - GOOGLE_CLOUD_PROJECT (required)
      - GOOGLE_CLOUD_LOCATION (optional, defaults to "europe-west1")

    Reference: https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai.md
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        pytest.skip("GOOGLE_CLOUD_PROJECT not set")

    return AnthropicVertex(
        project_id=project_id,
        region=os.environ.get("GOOGLE_CLOUD_LOCATION", "europe-west1"),
    )
