import random
import json
from locust import HttpUser, task, between, events


class LinguaBridgeUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(7)
    def send_text_message(self):
        payload = {
            "update_id": random.randint(10000, 99999),
            "message": {
                "message_id": f"msg_{random.randint(1000, 9999)}",
                "chat_id": "chat_test_123",
                "from": {"user_id": "user_456", "username": "test_user"},
                "timestamp": 1710000000,
                "text": "Салом, мен патентимни узайтирмоқчиман."
            }
        }

        with self.client.post("/webhooks/max/webhook", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(3)
    def send_voice_message(self):
        payload = {
            "update_id": random.randint(10000, 99999),
            "message": {
                "message_id": f"msg_{random.randint(1000, 9999)}",
                "chat": {"id": 123456},
                "from": {"id": 456, "username": "voice_user"},
                "date": 1710000000,
                "voice": {
                    "file_id": "voice_mock_123",
                    "file_unique_id": "unique_123",
                    "duration": 5,
                    "file_size": 45000
                }
            }
        }

        with self.client.post("/webhooks/telegram/webhook", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("🚀 Нагрузочное тестирование LinguaBridge началось!")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("🏁 Тестирование завершено. Проверьте метрики в Prometheus/Grafana.")
