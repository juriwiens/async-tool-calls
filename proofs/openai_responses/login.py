"""Login to OpenAI Codex API via ChatGPT subscription.

Supports two flows:
  --browser  (default) PKCE flow with local redirect server
  --device   Device code flow (may be disabled by workspace admins)

Saves tokens to .codex_tokens.json for use by the proof tests.

Usage:
    uv run python -m proofs.openai_responses.login
    uv run python -m proofs.openai_responses.login --device

Reference: https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/plugin/codex.ts
"""

import base64
import hashlib
import json
import os
import secrets
import sys
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

import httpx

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"
OAUTH_PORT = 1455
TOKEN_FILE = Path(__file__).parent.parent.parent / ".codex_tokens.json"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _generate_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def _extract_account_id(tokens: dict) -> str | None:
    for token_key in ("id_token", "access_token"):
        token = tokens.get(token_key, "")
        parts = token.split(".")
        if len(parts) != 3:
            continue
        try:
            payload = parts[1] + "=" * (-len(parts[1]) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            account_id = (
                claims.get("chatgpt_account_id")
                or (claims.get("https://api.openai.com/auth") or {}).get(
                    "chatgpt_account_id"
                )
                or (claims.get("organizations") or [{}])[0].get("id")
            )
            if account_id:
                return account_id
        except Exception:
            continue
    return None


def _save_tokens(tokens: dict):
    account_id = _extract_account_id(tokens)
    token_data = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_at": time.time() + tokens.get("expires_in", 3600),
        "account_id": account_id,
    }
    TOKEN_FILE.write_text(json.dumps(token_data, indent=2))
    print(f"\nLogin successful!")
    print(f"  Account ID: {account_id or 'unknown'}")
    print(f"  Tokens saved to: {TOKEN_FILE}")


# ── Browser PKCE Flow ─────────────────────────────────────────────────── #

def login_browser():
    """PKCE flow: opens browser, receives callback on localhost."""
    verifier, challenge = _generate_pkce()
    state = _b64url(secrets.token_bytes(32))
    redirect_uri = f"http://localhost:{OAUTH_PORT}/auth/callback"

    auth_url = (
        f"{ISSUER}/oauth/authorize?"
        + urlencode({
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": "openid profile email offline_access",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": "opencode",
        })
    )

    # Callback handler
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/auth/callback":
                params = parse_qs(parsed.query)
                received["code"] = params.get("code", [None])[0]
                received["state"] = params.get("state", [None])[0]
                received["error"] = params.get("error", [None])[0]

                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                if received.get("error"):
                    self.wfile.write(b"<h1>Authorization Failed</h1><p>You can close this window.</p>")
                else:
                    self.wfile.write(b"<h1>Authorization Successful</h1><p>You can close this window.</p>")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass  # silence logs

    print("Starting local server on port", OAUTH_PORT)
    print("Opening browser for OpenAI login...\n")
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", OAUTH_PORT), Handler)
    server.timeout = 300  # 5 min timeout
    server.handle_request()
    server.server_close()

    if received.get("error"):
        print(f"Error: {received['error']}")
        sys.exit(1)

    if not received.get("code"):
        print("No authorization code received.")
        sys.exit(1)

    if received.get("state") != state:
        print("State mismatch — possible CSRF.")
        sys.exit(1)

    # Exchange code for tokens
    resp = httpx.post(
        f"{ISSUER}/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": received["code"],
            "redirect_uri": redirect_uri,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    _save_tokens(resp.json())


# ── Device Code Flow ──────────────────────────────────────────────────── #

def login_device():
    """Device code flow: user enters code in browser."""
    resp = httpx.post(
        f"{ISSUER}/api/accounts/deviceauth/usercode",
        json={"client_id": CLIENT_ID},
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    device_data = resp.json()

    device_auth_id = device_data["device_auth_id"]
    user_code = device_data["user_code"]
    interval = max(int(device_data.get("interval", 5)), 1)

    print(f"  1. Open: {ISSUER}/codex/device")
    print(f"  2. Enter code: {user_code}")
    print(f"\nWaiting for authorization...\n")

    while True:
        resp = httpx.post(
            f"{ISSUER}/api/accounts/deviceauth/token",
            json={
                "device_auth_id": device_auth_id,
                "user_code": user_code,
            },
            headers={"Content-Type": "application/json"},
        )

        if resp.status_code == 200:
            data = resp.json()
            break
        if resp.status_code not in (403, 404):
            print(f"Unexpected status: {resp.status_code} — {resp.text}")
            sys.exit(1)
        time.sleep(interval + 3)

    resp = httpx.post(
        f"{ISSUER}/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": data["authorization_code"],
            "redirect_uri": f"{ISSUER}/deviceauth/callback",
            "client_id": CLIENT_ID,
            "code_verifier": data["code_verifier"],
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp.raise_for_status()
    _save_tokens(resp.json())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--browser"
    if mode == "--device":
        login_device()
    else:
        login_browser()
