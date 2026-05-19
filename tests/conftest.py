"""Pytest configuration."""
# ruff: noqa: D102,D107

from __future__ import annotations

import asyncio
from typing import Any

import fastapi.testclient
import httpx
import pytest


class _ASGITestClient:
    """Small synchronous client backed by httpx ASGITransport."""

    __test__ = False

    def __init__(self, app: Any) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async def _send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
                return await c.request(method, url, **kwargs)

        return asyncio.run(_send())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)


fastapi.testclient.TestClient = _ASGITestClient


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Temporary cache directory."""
    return tmp_path / "cache"


@pytest.fixture
def sample_data():
    """Sample test data."""
    return {
        "url": "https://example.com/data",
        "data": {"key": "value"},
    }
