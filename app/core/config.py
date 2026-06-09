from functools import lru_cache
from typing import Literal

from pydantic import AnyUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Централизованная конфигурация приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    # === Application ===
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    app_name: str = Field(default="LinguaBridge AI Mediator")
    app_version: str = Field(default="0.1.0")
    app_debug: bool = Field(default=False)
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000, ge=1, le=65535)

    # === PostgreSQL ===
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="linguabridge")
    postgres_password: SecretStr = Field(default=SecretStr("linguabridge"))
    postgres_db: str = Field(default="linguabridge")

    @property
    def postgres_dsn(self) -> str:
        pwd = self.postgres_password.get_secret_value()
        return f"postgresql+asyncpg://{self.postgres_user}:{pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_dsn_sync(self) -> str:
        """Для Alembic миграций (синхронный драйвер)."""
        pwd = self.postgres_password.get_secret_value()
        return f"postgresql://{self.postgres_user}:{pwd}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # === Redis ===
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: SecretStr = Field(default=SecretStr(""))
    redis_db: int = Field(default=0)

    @property
    def redis_url(self) -> str:
        pwd = self.redis_password.get_secret_value()
        auth = f":{pwd}@" if pwd else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # === RabbitMQ ===
    rabbitmq_host: str = Field(default="localhost")
    rabbitmq_port: int = Field(default=5672)
    rabbitmq_user: str = Field(default="guest")
    rabbitmq_password: SecretStr = Field(default=SecretStr("guest"))
    rabbitmq_vhost: str = Field(default="/")

    # Очереди
    rabbitmq_queue_text: str = Field(default="linguabridge.translate.text")
    rabbitmq_queue_voice: str = Field(default="linguabridge.translate.voice")
    rabbitmq_queue_dlq: str = Field(default="linguabridge.translate.dlq")
    rabbitmq_exchange: str = Field(default="linguabridge.exchange")

    @property
    def rabbitmq_url(self) -> str:
        pwd = self.rabbitmq_password.get_secret_value()
        return f"amqp://{self.rabbitmq_user}:{pwd}@{self.rabbitmq_host}:{self.rabbitmq_port}/{self.rabbitmq_vhost}"

    # === MinIO (S3-compatible) ===
    minio_endpoint: str = Field(default="http://localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: SecretStr = Field(default=SecretStr("minioadmin"))
    minio_bucket: str = Field(default="linguabridge-media")
    minio_region: str = Field(default="us-east-1")
    minio_audio_retention_days: int = Field(default=7, description="Lifecycle: удаление аудио через N дней (152-ФЗ)")

    # === Channel: MAX ===
    max_bot_token: SecretStr = Field(default=SecretStr(""))
    max_webhook_secret: SecretStr = Field(default=SecretStr(""))
    max_api_base_url: AnyUrl = Field(default="https://botapi.max.ru")  # Актуальный URL из dev.max.ru
    max_rate_limit_per_second: int = Field(default=30)

    # === Channel: VK ===
    vk_bot_token: SecretStr = Field(default=SecretStr(""))
    vk_confirmation_token: SecretStr = Field(default=SecretStr(""))
    vk_api_version: str = Field(default="5.199")

    # === Channel: Telegram ===
    telegram_bot_token: SecretStr = Field(default=SecretStr(""))
    telegram_webhook_secret: SecretStr = Field(default=SecretStr(""))

    # === ML Models ===
    fasttext_model_path: str = Field(default="./models/lid.176.bin")
    nllb_model_name: str = Field(default="facebook/nllb-200-distilled-600M")
    whisper_model_size: str = Field(
        default="large-v3-turbo",
        description="Размер модели: tiny, base, small, medium, large-v3, large-v3-turbo"
    )
    whisper_compute_type: str = Field(
        default="int8",
        description="int8 (рекомендуется для CPU/GPU), float16 (только GPU), default"
    )
    whisper_vad_filter: bool = Field(default=True, description="Включить фильтрацию голоса (Voice Activity Detection)")
    min_asr_confidence: float = Field(default=0.6, description="Минимальная уверенность ASR для принятия текста")

    # === NLLB Translation Engine ===
    nllb_device: str = Field(default="auto", description="cuda, cpu или auto")
    nllb_quantize_8bit: bool = Field(default=True, description="Использовать 8-битное квантование для экономии памяти")
    nllb_max_length: int = Field(default=512, description="Максимальная длина генерации в токенах")

    # === amoCRM Integration ===
    amocrm_subdomain: str = Field(default="your-company", description="Поддомен amoCRM (без .amocrm.ru)")
    amocrm_access_token: SecretStr = Field(default=SecretStr(""))
    amocrm_custom_field_user_id: int = Field(default=123456)
    amocrm_request_timeout: float = Field(default=5.0, description="Таймаут запросов к amoCRM")
    amocrm_cb_fail_max: int = Field(default=3, description="Кол-во ошибок подряд для размыкания цепи")
    amocrm_cb_reset_timeout: int = Field(default=60, description="Секунд до попытки восстановления цепи")

    # === Fraud Detection ===
    fraud_score_threshold: float = Field(default=0.7, description="Порог срабатывания алерта о мошенничестве (0.0 - 1.0)")

    # === Audio Pre-processing ===
    audio_enable_noise_reduction: bool = Field(default=True, description="Включить фильтрацию шума перед ASR")
    audio_highpass_freq: int = Field(default=200, description="Частота среза низких частот (удаление гула)")
    audio_lowpass_freq: int = Field(default=8000, description="Частота среза высоких частот (удаление шипения)")

    # === Silero TTS Engine ===
    silero_tts_enabled: bool = Field(default=True)
    silero_tts_default_speaker: str = Field(default="kseniya", description="kseniya (RU), bryan (EN), thorsten (DE)")
    silero_tts_sample_rate: int = Field(default=24000, description="Частота дискретизации выходного wav")
    tts_fallback_provider: str = Field(default="yandex", description="yandex, local, none")

    # === Translation Settings ===
    supported_languages: list[str] = Field(
        default=["tg", "uz", "ky", "zh", "en", "ru"],
        description="ISO 639-1 codes supported languages"
    )
    translation_latency_target_ms: int = Field(default=1500)
    voice_latency_target_ms: int = Field(default=4000)

    # === Observability ===
    otlp_endpoint: str = Field(default="http://localhost:4317")
    otlp_service_name: str = Field(default="linguabridge")


@lru_cache
def get_settings() -> Settings:
    """Кэшированный доступ к конфигурации."""
    return Settings()


settings = get_settings()
