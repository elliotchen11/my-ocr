"""
azure_authentication.py

Azure Entra ID (Azure AD) authentication for Streamlit using MSAL.
Flow: Authorization Code → ID token returned to Streamlit session state.

Setup (Azure Portal):
  1. Go to Azure Portal > Entra ID > App registrations > New registration
  2. Set Redirect URI to: http://localhost:8501  (or your Streamlit URL)
  3. Under "Certificates & secrets", create a Client secret
  4. Under "API permissions", add "openid", "profile", "email" (Microsoft Graph delegated)
  5. Copy Tenant ID, Client ID, and Client Secret into config below or via env vars

Install:
  pip install msal
"""

from __future__ import annotations

import os
import base64
import json
from typing import Optional, Dict, Any

import msal
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration — override via environment variables or pass directly
# ---------------------------------------------------------------------------

# Scopes required for ID token (user identity only)
SCOPES = ["User.Read"]

# Server-side flow store — survives browser redirects (module stays in memory)
_pending_flows: Dict[str, dict] = {}


def _cfg() -> dict:
    """Read config from env vars at call time (never cached at import)."""
    tenant_id     = os.getenv("AZURE_TENANT_ID", "")
    client_id     = os.getenv("AZURE_CLIENT_ID", "")
    client_secret = os.getenv("AZURE_CLIENT_SECRET", "")
    redirect_uri  = os.getenv("AZURE_REDIRECT_URI", "http://localhost:8501")

    if not tenant_id or not client_id or not client_secret:
        missing = [k for k, v in {
            "AZURE_TENANT_ID": tenant_id,
            "AZURE_CLIENT_ID": client_id,
            "AZURE_CLIENT_SECRET": client_secret,
        }.items() if not v]
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return {
        "tenant_id":     tenant_id,
        "client_id":     client_id,
        "client_secret": client_secret,
        "redirect_uri":  redirect_uri,
        "authority":     f"https://login.microsoftonline.com/{tenant_id}",
    }


# ---------------------------------------------------------------------------
# MSAL app (confidential — uses client secret)
# ---------------------------------------------------------------------------

def _get_msal_app(cache: Optional[msal.SerializableTokenCache] = None) -> msal.ConfidentialClientApplication:
    cfg = _cfg()
    return msal.ConfidentialClientApplication(
        client_id=cfg["client_id"],
        client_credential=cfg["client_secret"],
        authority=cfg["authority"],
        token_cache=cache,
    )


# ---------------------------------------------------------------------------
# Token cache — stored in Streamlit session state
# ---------------------------------------------------------------------------

def _load_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if "msal_token_cache" in st.session_state:
        cache.deserialize(st.session_state["msal_token_cache"])
    return cache


def _save_cache(cache: msal.SerializableTokenCache) -> None:
    if cache.has_state_changed:
        st.session_state["msal_token_cache"] = cache.serialize()


# ---------------------------------------------------------------------------
# Core auth helpers
# ---------------------------------------------------------------------------

def initiate_auth_flow() -> str:
    """
    Start the MSAL auth code flow (handles PKCE, state, nonce automatically).
    Stores the flow in a module-level dict keyed by state so it survives
    the browser redirect to Microsoft and back.
    """
    cfg = _cfg()
    app = _get_msal_app()
    flow = app.initiate_auth_code_flow(
        scopes=SCOPES,
        redirect_uri=cfg["redirect_uri"],
    )
    _pending_flows[flow["state"]] = flow
    return flow["auth_uri"]


def exchange_code_for_token(auth_response: dict) -> Optional[Dict[str, Any]]:
    """
    Complete the auth code flow using the query params from the redirect URL.
    Looks up the original flow by the echoed `state` parameter.
    """
    state = auth_response.get("state", "")
    flow = _pending_flows.pop(state, None)

    if not flow:
        st.error("Auth session expired or not found. Please sign in again.")
        return None

    cache = _load_cache()
    app = _get_msal_app(cache)

    result = app.acquire_token_by_auth_code_flow(
        auth_code_flow=flow,
        auth_response=auth_response,
    )

    _save_cache(cache)

    if "error" in result:
        st.error(f"Authentication error: {result.get('error_description', result['error'])}")
        return None

    return result


def get_cached_token() -> Optional[Dict[str, Any]]:
    """
    Try to retrieve a valid token silently from the cache.
    Returns the token response dict or None if no cached account.
    """
    cache = _load_cache()
    app = _get_msal_app(cache)

    accounts = app.get_accounts()
    if not accounts:
        return None

    result = app.acquire_token_silent(scopes=SCOPES, account=accounts[0])
    _save_cache(cache)

    if result and "error" not in result:
        return result
    return None


def decode_id_token_claims(id_token: str) -> Dict[str, Any]:
    """
    Decode the JWT ID token payload (no signature verification —
    MSAL already validated it during acquisition).
    """
    try:
        payload_b64 = id_token.split(".")[1]
        # Add padding if needed
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload)
    except Exception:
        return {}


def logout() -> None:
    """Clear the token cache and session state."""
    for key in ("msal_token_cache", "azure_user"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Streamlit integration — call this at the top of your app page
# ---------------------------------------------------------------------------

def authenticate_streamlit_user() -> Optional[Dict[str, Any]]:
    """
    Full auth code flow for Streamlit.

    - If the user is already authenticated (token in session), returns user claims.
    - If a ?code= query param is present (redirect from Microsoft), exchanges it.
    - Otherwise shows a "Sign in with Microsoft" button.

    Returns:
        dict of ID token claims (sub, name, email, preferred_username, etc.)
        or None if not yet authenticated.

    Usage:
        user = authenticate_streamlit_user()
        if user is None:
            st.stop()
        st.write(f"Hello, {user.get('name')}")
    """
    # 1. Already have a valid cached token?
    if "azure_user" in st.session_state:
        return st.session_state["azure_user"]

    token = get_cached_token()
    if token:
        claims = decode_id_token_claims(token.get("id_token", ""))
        st.session_state["azure_user"] = claims
        return claims

    # 2. Coming back from Microsoft with an auth code?
    params = dict(st.query_params)
    if "code" in params:
        token = exchange_code_for_token(params)
        st.query_params.clear()

        if token:
            claims = decode_id_token_claims(token.get("id_token", ""))
            st.session_state["azure_user"] = claims
            st.rerun()

        return None

    # 3. Not authenticated — show login button
    auth_url = initiate_auth_flow()
    st.markdown("## Sign in required")
    st.markdown(
        f'<a href="{auth_url}" target="_self">'
        f'<button style="background:#0078d4;color:white;border:none;padding:10px 20px;'
        f'border-radius:4px;cursor:pointer;font-size:14px;">Sign in with Microsoft</button>'
        f"</a>",
        unsafe_allow_html=True,
    )
    return None
