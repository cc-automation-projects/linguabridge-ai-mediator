from aiobotocore.config import AioConfig
from aiobotocore.session import get_session
from botocore.exceptions import ClientError

from app.core.config import settings


class S3Client:
    """Асинхронный клиент для S3-совместимого хранилища (MinIO)."""

    def __init__(self) -> None:
        self._session = get_session()
        self._config = AioConfig(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=5,
            read_timeout=30,
        )

    def _client_context(self):
        return self._session.create_client(
            "s3",
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key.get_secret_value(),
            region_name=settings.minio_region,
            config=self._config,
        )

    async def ensure_bucket_exists(self) -> None:
        """Создание бакета, если он не существует."""
        async with self._client_context() as client:
            try:
                await client.head_bucket(Bucket=settings.minio_bucket)
            except ClientError:
                await client.create_bucket(Bucket=settings.minio_bucket)

    async def configure_lifecycle(self) -> None:
        """
        Настройка автоматического удаления объектов через N дней (152-ФЗ).
        """
        lifecycle_config = {
            "Rules": [
                {
                    "ID": "auto-delete-old-audio",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},  # Применяется ко всем объектам
                    "Expiration": {"Days": settings.minio_audio_retention_days},
                }
            ]
        }
        async with self._client_context() as client:
            try:
                await client.put_bucket_lifecycle_configuration(
                    Bucket=settings.minio_bucket,
                    LifecycleConfiguration=lifecycle_config,
                )
            except ClientError as e:
                # Некоторые S3-совместимые хранилища не поддерживают lifecycle
                # Логируем warning, но не падаем
                print(f"Warning: Lifecycle policy not applied: {e}")

    async def upload_file(
        self,
        key: str,
        body: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict | None = None,
    ) -> str:
        """Загрузка файла в бакет."""
        async with self._client_context() as client:
            await client.put_object(
                Bucket=settings.minio_bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata=metadata or {},
            )
        return key

    async def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Генерация временной ссылки для скачивания."""
        async with self._client_context() as client:
            url = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.minio_bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url

    async def download_file(self, key: str) -> bytes:
        """Скачивание файла из бакета."""
        async with self._client_context() as client:
            response = await client.get_object(
                Bucket=settings.minio_bucket,
                Key=key,
            )
            async with response["Body"] as stream:
                return await stream.read()

    async def delete_file(self, key: str) -> None:
        """Удаление файла."""
        async with self._client_context() as client:
            await client.delete_object(Bucket=settings.minio_bucket, Key=key)


s3_client = S3Client()
