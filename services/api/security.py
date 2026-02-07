"""API key security with scope enforcement."""
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from services.api.config import get_api_keys

API_KEY_NAME = "X-GridPulse-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    api_keys = get_api_keys()
    if api_key_header in api_keys:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Could not validate credentials",
    )


def verify_scope(required_scope: str, api_key: str) -> None:
    user_scopes = get_api_keys().get(api_key, [])
    if required_scope not in user_scopes:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing required scope: {required_scope}",
        )
