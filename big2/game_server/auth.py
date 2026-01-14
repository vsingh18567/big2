"""API key authentication for the game server."""

import os
from typing import Annotated

from fastapi import Header, HTTPException, status

# Default API key (can be overridden via environment variable)
DEFAULT_API_KEY = os.getenv("BIG2_API_KEY", "dev-api-key-changeme")


async def verify_api_key(x_api_key: Annotated[str, Header()]) -> str:
    """
    Verify the API key from request headers.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The API key if valid

    Raises:
        HTTPException: 401 if API key is invalid
    """
    if x_api_key != DEFAULT_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key
