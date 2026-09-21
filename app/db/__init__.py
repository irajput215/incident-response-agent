"""Database layer: connections, versioned migrations, and repositories."""

from app.db.connection import (
    DatabaseUnavailable,
    close_pools,
    connect,
    database_available,
    get_pool,
)
from app.db.repository import IncidentRepository, new_id
from app.db.schema import LATEST_VERSION, ensure_schema, migrate

__all__ = [
    "LATEST_VERSION",
    "DatabaseUnavailable",
    "IncidentRepository",
    "close_pools",
    "connect",
    "database_available",
    "ensure_schema",
    "get_pool",
    "migrate",
    "new_id",
]
