from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""
    pass


def _create_engine() -> AsyncEngine:
    """Создание движка с оптимальными параметрами пула соединений."""
    return create_async_engine(
        settings.postgres_dsn,
        echo=settings.app_debug,
        pool_size=20,           # Базовый размер пула
        max_overflow=10,        # Максимум дополнительных соединений
        pool_timeout=30,        # Таймаут ожидания соединения (сек)
        pool_recycle=3600,      # Переподключение каждый час
        pool_pre_ping=True,     # Проверка "живости" соединения перед использованием
    )


engine: AsyncEngine = _create_engine()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Зависимость для FastAPI endpoint-ов.
    Автоматически закрывает сессию и откатывает транзакцию при ошибке.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Создание таблиц (для dev/test окружения)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Корректное закрытие пула соединений."""
    await engine.dispose()
