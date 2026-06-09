from unittest.mock import patch

import httpx
import pybreaker
import pytest
import respx

from app.core.models import Channel
from app.integrations.amocrm_client import amo_circuit_breaker, amocrm_client


@pytest.fixture(autouse=True)
def reset_breaker():
    amo_circuit_breaker.reset()
    yield


@respx.mock
@pytest.mark.asyncio
async def test_find_or_create_lead_success():
    respx.post("https://test-company.amocrm.ru/api/v4/leads/complex/search").respond(
        json={"_embedded": {"leads": []}}
    )
    respx.post("https://test-company.amocrm.ru/api/v4/leads").respond(
        json={"_embedded": {"leads": [{"id": 98765}]}}
    )

    lead_id = await amocrm_client.find_or_create_lead("user_123", Channel.MAX, "Test User")
    assert lead_id == 98765


@respx.mock
@pytest.mark.asyncio
async def test_amo_fake_200_error():
    respx.post("https://test-company.amocrm.ru/api/v4/leads").respond(
        json={"status": "error", "title": "Invalid token", "detail": "Token expired"}
    )

    with pytest.raises(ValueError, match="amoCRM API Error: Invalid token"):
        await amocrm_client.find_or_create_lead("user_123", Channel.MAX, "Test User")


@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_repeated_failures():
    for _ in range(3):
        with patch.object(httpx.Client, 'request', side_effect=httpx.ConnectError("Connection refused")):
            try:
                await amocrm_client.find_or_create_lead("user_123", Channel.MAX, "Test")
            except httpx.ConnectError:
                pass

    with pytest.raises(pybreaker.CircuitBreakerError):
        await amocrm_client.find_or_create_lead("user_123", Channel.MAX, "Test")
