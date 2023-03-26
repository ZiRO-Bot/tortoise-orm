import asyncio
import os
import sqlite3
from functools import wraps
from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import asqlite
from pypika import SQLLiteQuery

from tortoise.backends.base.client import (
    BaseDBAsyncClient,
    BaseTransactionWrapper,
    Capabilities,
    ConnectionWrapper,
    NestedTransactionPooledContext,
    PoolConnectionWrapper,
    TransactionContext,
    TransactionContextPooled,
)
from tortoise.backends.asqlite.executor import SqliteExecutor
from tortoise.backends.asqlite.schema_generator import SqliteSchemaGenerator
from tortoise.exceptions import (
    IntegrityError,
    OperationalError,
    TransactionManagementError,
)

FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def translate_exceptions(func: F) -> F:
    @wraps(func)
    async def translate_exceptions_(self, query, *args):
        try:
            return await func(self, query, *args)
        except sqlite3.OperationalError as exc:
            raise OperationalError(exc)
        except sqlite3.IntegrityError as exc:
            raise IntegrityError(exc)

    return translate_exceptions_  # type: ignore


class SqliteClient(BaseDBAsyncClient):
    executor_class = SqliteExecutor
    query_class = SQLLiteQuery
    schema_generator = SqliteSchemaGenerator
    capabilities = Capabilities(
        "sqlite", daemon=False, requires_limit=True, inline_comment=True, support_for_update=False
    )
    _pool: Optional[asqlite.Pool] = None
    _connection: asqlite.Connection

    def __init__(self, file_path: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.filename = file_path

    async def create_connection(self, with_db: bool) -> None:
        if not self._pool:
            self._pool = await self.create_pool()

    async def create_pool(self, **kwargs) -> asqlite.Pool:
        return await asqlite.create_pool(self.filename, **kwargs)

    async def close(self) -> None:
        if self._pool:
            try:
                await self._pool.close()
            except Exception:
                pass  # asqlite already call terminate()
            self.log.debug(
                "Closed connection pool %s with params: filename=%s",
                self._pool,
                self.filename,
            )
            self._pool = None

    async def db_create(self) -> None:
        # DB's are automatically created once accessed
        pass

    async def db_delete(self) -> None:
        await self.close()
        try:
            os.remove(self.filename)
        except FileNotFoundError:  # pragma: nocoverage
            pass
        except OSError as e:
            if e.errno != 22:  # fix: "sqlite://:memory:" in Windows
                raise e

    def acquire_connection(self) -> Union[PoolConnectionWrapper, ConnectionWrapper]:
        return PoolConnectionWrapper(self)

    def _in_transaction(self) -> "TransactionContext":
        return TransactionContextPooled(TransactionWrapper(self))

    @translate_exceptions
    async def execute_insert(self, query: str, values: list) -> int:
        async with self.acquire_connection() as connection:
            self.log.debug("%s: %s", query, values)
            cursor: asqlite.Cursor = await connection.execute(query, values)
            await cursor.execute("SELECT last_insert_rowid()")
            return (await cursor.fetchone())[0]

    @translate_exceptions
    async def execute_many(self, query: str, values: List[list]) -> None:
        async with self.acquire_connection() as connection:
            self.log.debug("%s: %s", query, values)
            # This code is only ever called in AUTOCOMMIT mode
            await connection.execute("BEGIN")
            try:
                await connection.executemany(query, values)
            except Exception:
                await connection.rollback()
                raise
            else:
                await connection.commit()

    @translate_exceptions
    async def execute_query(
        self, query: str, values: Optional[list] = None
    ) -> Tuple[int, Sequence[sqlite3.Row]]:
        query = query.replace("\x00", "'||CHAR(0)||'")
        async with self.acquire_connection() as connection:
            self.log.debug("%s: %s", query, values)
            start = connection.total_changes
            cursor: asqlite.Cursor = await connection.execute(query, values)
            rows = await cursor.fetchall()
            return (connection.total_changes - start) or len(rows), rows

    @translate_exceptions
    async def execute_query_dict(self, query: str, values: Optional[list] = None) -> List[dict]:
        query = query.replace("\x00", "'||CHAR(0)||'")
        async with self.acquire_connection() as connection:
            self.log.debug("%s: %s", query, values)
            cursor: asqlite.Cursor = await connection.execute(query, values)
            rows = await cursor.fetchall()
            return list(map(dict, rows))

    @translate_exceptions
    async def execute_script(self, query: str) -> None:
        async with self.acquire_connection() as connection:
            self.log.debug(query)
            await connection.executescript(query)


class TransactionWrapper(SqliteClient, BaseTransactionWrapper):
    def __init__(self, connection: SqliteClient) -> None:
        self._connection: asqlite.Connection = connection._connection
        self._lock = asyncio.Lock()
        self._trxlock = asyncio.Lock()
        self.log = connection.log
        self.connection_name = connection.connection_name
        self.transaction: asqlite.Transaction = None  # type: ignore
        self._finalized = False
        self.fetch_inserted = connection.fetch_inserted
        self._parent: SqliteClient = connection

    def _in_transaction(self) -> "TransactionContext":
        return NestedTransactionPooledContext(self)

    def acquire_connection(self) -> ConnectionWrapper:
        return ConnectionWrapper(self._lock, self)

    @translate_exceptions
    async def execute_many(self, query: str, values: List[list]) -> None:
        async with self.acquire_connection() as connection:
            self.log.debug("%s: %s", query, values)
            # Already within transaction, so ideal for performance
            await connection.executemany(query, values)

    @translate_exceptions
    async def start(self) -> None:
        self.transaction = self._connection.transaction()
        await self.transaction.start()

    async def commit(self) -> None:
        if self._finalized:
            raise TransactionManagementError("Transaction already finalised")
        await self.transaction.commit()
        self._finalized = True

    async def rollback(self) -> None:
        if self._finalized:
            raise TransactionManagementError("Transaction already finalised")
        await self.transaction.rollback()
        self._finalized = True
