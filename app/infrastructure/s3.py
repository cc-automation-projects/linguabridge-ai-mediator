
from app.core.logger import logger
from app.core.s3 import s3_client


class S3Service:
    async def upload_file(self, key: str, body: bytes, content_type: str = "application/octet-stream") -> bool:
        try:
            await s3_client.upload_file(key, body, content_type=content_type)
            return True
        except Exception as e:
            logger.error("s3_upload_failed", key=key, error=str(e), exc_info=True)
            return False

    async def download_file(self, key: str) -> bytes | None:
        try:
            return await s3_client.download_file(key)
        except Exception as e:
            logger.error("s3_download_failed", key=key, error=str(e), exc_info=True)
            return None


s3_service = S3Service()
