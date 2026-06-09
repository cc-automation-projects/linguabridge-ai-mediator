from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Channel(str, Enum):
    MAX = "max"
    VK = "vk"
    TELEGRAM = "telegram"


class MediaType(str, Enum):
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"


class Direction(str, Enum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"


class ConversationTurn(BaseModel):
    role: str
    original_lang: str
    original_text: str
    translated_text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationContext(BaseModel):
    user_id: str
    channel: Channel
    turns: list[ConversationTurn] = Field(default_factory=list, max_length=10)
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)
        if len(self.turns) > 10:
            self.turns = self.turns[-10:]
        self.last_activity_at = datetime.now(UTC)


class IncomingMessage(BaseModel):
    message_id: str
    channel: Channel
    user_id: str
    user_display_name: str | None = None
    chat_id: str | None = None
    text: str | None = None
    audio_s3_key: str | None = None
    media_type: MediaType | None = None
    direction: Direction = Direction.INCOMING
    detected_lang: str | None = None
    lang_confidence: float | None = None
    masked_text: str | None = None
    translated_text: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
