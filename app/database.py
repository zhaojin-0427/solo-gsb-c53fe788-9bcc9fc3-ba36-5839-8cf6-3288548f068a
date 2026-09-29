"""Database engine / session management."""
from __future__ import annotations

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from . import config

_is_sqlite = config.DATABASE_URL.startswith("sqlite")
connect_args = (
    {"check_same_thread": False, "timeout": 30.0} if _is_sqlite else {}
)
engine = create_engine(config.DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        # WAL + busy_timeout let concurrent creators/activators block on the
        # unique constraints instead of failing with "database is locked".
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


class Base(DeclarativeBase):
    pass


def run_lightweight_migrations(bind=None) -> None:
    """Schema evolutions for deployments whose tables predate a new column.

    ``create_all`` never alters existing tables, so additions are applied
    here.  The focal-mechanism dedup-rule marker is added with default 1
    (legacy rule), so every already-persisted report is flagged for lazy
    repair on the next repeat submission or fetch by id; brand-new tables
    are created by the ORM metadata with a Python default of 2.
    """
    bind = bind or engine
    inspector = inspect(bind)
    if "focal_mechanism_inversions" not in inspector.get_table_names():
        return  # created from scratch with all current columns
    columns = {c["name"] for c in inspector.get_columns("focal_mechanism_inversions")}
    if "dedup_rule_version" in columns:
        return
    with bind.begin() as conn:
        # The column is added with default 1: every already-persisted row is a
        # legacy (M == -M) report that needs lazy repair.  Fresh tables are
        # created by the ORM metadata with server_default 2 instead.
        conn.execute(
            text(
                "ALTER TABLE focal_mechanism_inversions "
                "ADD COLUMN dedup_rule_version INTEGER "
                "NOT NULL DEFAULT 1"
            )
        )


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
