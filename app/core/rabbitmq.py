
import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractQueue, AbstractRobustConnection

from app.core.config import settings


class RabbitMQClient:
    """Асинхронный клиент RabbitMQ с robust connection."""

    def __init__(self) -> None:
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._exchange: AbstractExchange | None = None
        self._queues: dict[str, AbstractQueue] = {}

    async def connect(self) -> None:
        """Установка соединения и декларация инфраструктуры очередей."""
        self._connection = await aio_pika.connect_robust(settings.rabbitmq_url)
        self._channel = await self._connection.channel()

        # QoS: каждый воркер берет не более 1 сообщения
        await self._channel.set_qos(prefetch_count=1)

        # Декларируем exchange (topic type для гибкой маршрутизации)
        self._exchange = await self._channel.declare_exchange(
            settings.rabbitmq_exchange,
            type=aio_pika.ExchangeType.TOPIC,
            durable=True,
        )

        # Декларируем DLQ (Dead Letter Queue)
        dlq = await self._channel.declare_queue(
            settings.rabbitmq_queue_dlq,
            durable=True,
        )
        self._queues["dlq"] = dlq

        # Декларируем основную очередь для текстовых сообщений (высокий приоритет)
        text_queue = await self._channel.declare_queue(
            settings.rabbitmq_queue_text,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": settings.rabbitmq_queue_dlq,
                "x-max-priority": 10,  # Поддержка приоритетов
            },
        )
        await text_queue.bind(self._exchange, routing_key="translate.text")
        self._queues["text"] = text_queue

        # Декларируем очередь для голосовых сообщений (низкий приоритет, тяжелее)
        voice_queue = await self._channel.declare_queue(
            settings.rabbitmq_queue_voice,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": settings.rabbitmq_queue_dlq,
                "x-max-priority": 5,
            },
        )
        await voice_queue.bind(self._exchange, routing_key="translate.voice")
        self._queues["voice"] = voice_queue

    @property
    def channel(self) -> AbstractChannel:
        if self._channel is None:
            raise RuntimeError("RabbitMQ not connected")
        return self._channel

    @property
    def exchange(self) -> AbstractExchange:
        if self._exchange is None:
            raise RuntimeError("RabbitMQ not connected")
        return self._exchange

    def get_queue(self, name: str) -> AbstractQueue:
        if name not in self._queues:
            raise KeyError(f"Queue '{name}' not found")
        return self._queues[name]

    async def close(self) -> None:
        if self._connection:
            await self._connection.close()


rabbitmq_client = RabbitMQClient()
