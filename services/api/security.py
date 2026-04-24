"""API key security with scope enforcement."""
from __future__ import annotations

from collections.abc import Iterable

from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from services.api.config import get_api_keys, is_auth_disabled_for_tests

API_KEY_NAME = "X-ORIUS-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if is_auth_disabled_for_tests():
        return "test-auth-disabled"
    api_keys = get_api_keys()
    if api_key_header in api_keys:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )


def verify_scope(required_scope: str, api_key: str) -> None:
    if is_auth_disabled_for_tests():
        return
    user_scopes = get_api_keys().get(api_key, [])
    if required_scope not in user_scopes:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing required scope: {required_scope}",
        )


def verify_any_scope(required_scopes: Iterable[str], api_key: str) -> None:
    """Authorize when an API key has at least one required scope."""
    if is_auth_disabled_for_tests():
        return
    required = set(required_scopes)
    user_scopes = set(get_api_keys().get(api_key, []))
    if required.isdisjoint(user_scopes):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing one of required scopes: {', '.join(sorted(required))}",
        )
