import contextlib
from typing import Any, AsyncIterator

from sqlalchemy import event, NullPool
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from geminiplayground.utils import Singleton
from .config import settings
from .registry import mapper_registry


class SessionManager(metaclass=Singleton):
    """
    Database session manager
    """

    def __init__(
            self,
            url: str,
            enable_foreign_keys: bool = True,
            autocommit: bool = False,
            autflush: bool = False,
            expire_on_commit: bool = False,
            engine_kwargs: Any = None,
    ):
        if engine_kwargs is None:
            engine_kwargs = {}

        self._engine = create_async_engine(url, **engine_kwargs)
        if enable_foreign_keys:
            @event.listens_for(self._engine.sync_engine, "connect")
            def _fk_pragma_on_connect(dbapi_con, con_record):
                dbapi_con.execute("pragma foreign_keys=ON")

        self._session_maker = async_sessionmaker(
            autocommit=autocommit,
            autoflush=autflush,
            expire_on_commit=expire_on_commit,
            bind=self._engine,
        )

    async def init(self, drop_all: bool = False):
        """
        Initialize the database
        """
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        async with self._engine.begin() as conn:
            if drop_all:
                await conn.run_sync(mapper_registry.metadata.drop_all)
            await conn.run_sync(mapper_registry.metadata.create_all)
        return self

    async def close(self):
        """
        Close the database
        """
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        await self._engine.dispose()
        self._engine = None
        self._session_maker = None

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """
        Connect to the database
        """
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")

        async with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """
        Create a new session
        """
        if self._session_maker is None:
            raise Exception("DatabaseSessionManager is not initialized")

        session = self._session_maker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


sessionmanager = SessionManager(
    url=settings.database_url,
    engine_kwargs={
        "echo": settings.echo_sql,
        "poolclass": NullPool,
        "pool_recycle": 3600,
        "future": True,
    },
)


async def get_db_session():
    """
    Get a database session
    """
    async with sessionmanager.session() as session:
        yield session
