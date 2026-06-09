import asyncio
from typing import Any
from urllib.parse import urljoin

import httpx
import pybreaker

from app.core.config import settings
from app.core.logger import logger
from app.core.models import Channel

amo_circuit_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.amocrm_cb_fail_max,
    reset_timeout=settings.amocrm_cb_reset_timeout,
    name="amocrm_api_breaker"
)


class AmoCRMClient:
    def __init__(self):
        self.base_url = f"https://{settings.amocrm_subdomain}.amocrm.ru/api/v4/"
        self.token = settings.amocrm_access_token.get_secret_value()

        self.client = httpx.AsyncClient(
            timeout=settings.amocrm_request_timeout,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        logger.info("amocrm_client_initialized_with_circuit_breaker")

    def _check_amo_error(self, response: httpx.Response) -> None:
        response.raise_for_status()

        try:
            data = response.json()
            if isinstance(data, dict) and data.get("status") == "error":
                raise ValueError(f"amoCRM API Error: {data.get('title')} - {data.get('detail')}")
        except httpx.JSONDecodeError:
            pass

    async def _make_request_async(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        url = urljoin(self.base_url, endpoint)

        def _sync_request() -> dict[str, Any]:
            @amo_circuit_breaker
            def _protected_request():
                with httpx.Client(
                    timeout=settings.amocrm_request_timeout,
                    headers=self.client.headers
                ) as sync_client:
                    response = sync_client.request(method, url, **kwargs)
                    self._check_amo_error(response)
                    return response.json()
            return _protected_request()

        try:
            result = await asyncio.to_thread(_sync_request)
            return result
        except pybreaker.CircuitBreakerError:
            logger.critical(
                "amocrm_circuit_breaker_open",
                msg="amoCRM недоступен или возвращает ошибки. Запрос отклонен для защиты системы."
            )
            raise
        except Exception as e:
            logger.error("amocrm_request_failed", endpoint=endpoint, error=str(e), exc_info=True)
            raise

    async def find_or_create_lead(self, user_id: str, channel: Channel, user_display_name: str | None) -> int:
        field_id = settings.amocrm_custom_field_user_id

        search_payload = {
            "query": {
                "custom_fields_values": [
                    {
                        "field_id": field_id,
                        "values": [{"value": user_id}]
                    }
                ]
            },
            "limit": 1
        }

        try:
            response = await self._make_request_async("POST", "leads/complex/search", json=search_payload)
            leads = response.get("_embedded", {}).get("leads", [])
            if leads:
                logger.info("amocrm_lead_found", lead_id=leads[0]["id"], user_id=user_id)
                return leads[0]["id"]
        except Exception as e:
            logger.warning("amocrm_search_failed_fallback_to_create", error=str(e))

        channel_name = channel.value.upper()
        new_lead_payload = {
            "name": f"Заявка из {channel_name}: {user_display_name or 'Неизвестный'}",
            "custom_fields_values": [
                {
                    "field_id": field_id,
                    "values": [{"value": user_id}]
                }
            ]
        }

        response = await self._make_request_async("POST", "leads", json=[new_lead_payload])
        lead_id = response["_embedded"]["leads"][0]["id"]
        logger.info("amocrm_lead_created", lead_id=lead_id, user_id=user_id)
        return lead_id

    async def add_note(self, lead_id: int, note_text: str) -> None:
        note_payload = {
            "note_type": "common",
            "params": {
                "text": note_text
            }
        }
        await self._make_request_async("POST", f"leads/{lead_id}/notes", json=[note_payload])
        logger.info("amocrm_note_added", lead_id=lead_id)

    async def update_tags(self, lead_id: int, tags_to_add: list[str]) -> None:
        update_payload = {
            "tags": [{"name": tag} for tag in tags_to_add]
        }
        await self._make_request_async("PATCH", f"leads/{lead_id}", json=update_payload)
        logger.info("amocrm_tags_updated", lead_id=lead_id, tags=tags_to_add)

    async def close(self):
        await self.client.aclose()


amocrm_client = AmoCRMClient()
