"""Daily Kite Connect login (tokens expire every morning around 06:00).

Option A - browser: set your Kite app's Redirect URL to <PUBLIC_URL>/auth/callback and click
"Login to Kite" in the web UI. The backend stores the token automatically.

Option B - terminal:
    python -m auth.login            # prints the login URL, then paste the redirected URL / request_token
    python -m auth.login --status   # show whether today's token is valid
"""
from __future__ import annotations

import argparse
import sys
from urllib.parse import parse_qs, urlparse

from core.kite import complete_login, kite_configured, login_url, token_status


def extract_request_token(text: str) -> str:
    text = text.strip()
    if text.startswith("http"):
        qs = parse_qs(urlparse(text).query)
        if "request_token" not in qs:
            raise SystemExit("No request_token in that URL")
        return qs["request_token"][0]
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--request-token", help="skip the prompt")
    args = ap.parse_args(argv)

    if args.status:
        st = token_status()
        print(f"valid={st['valid']} user={st['user']} saved={st['saved_at']} expires={st['expires_at']}")
        return 0 if st["valid"] else 1
    if not kite_configured():
        raise SystemExit("Set KITE_API_KEY and KITE_API_SECRET in .env")

    token = args.request_token
    if not token:
        print("1. Open this URL and log in to Kite:\n\n   " + login_url() + "\n")
        print("2. After login you are redirected to your Redirect URL. Paste that full URL")
        print("   (or just the request_token value) here:\n")
        token = input("> ")
    user = complete_login(extract_request_token(token))
    print(f"Logged in as {user.get('user_name')} ({user.get('user_id')}). Token saved; valid until ~06:00 tomorrow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
