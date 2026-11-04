# src/sandbox/executor.py
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

BLOCKED_SQL_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|VACUUM|REINDEX|PRAGMA)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ExecutionResult:
    status: str
    result: list[tuple[Any, ...]] | None = None
    error_message: str | None = None
    execution_time: float = 0.0


def is_readonly_query(sql: str) -> bool:
    """Validates that query is strictly read-only and free of DDL/DML side effects."""
    cleaned = re.sub(
        r"--.*?$|/\*.*?\*/", "", sql, flags=re.MULTILINE | re.DOTALL
    ).strip()
    if not cleaned:
        return False
    if BLOCKED_SQL_PATTERNS.search(cleaned):
        return False
    return cleaned.upper().startswith(("SELECT", "WITH"))


def execute_sql(
    db_path: str, sql: str, timeout_seconds: float = 10.0
) -> ExecutionResult:
    """Executes a SQL query against SQLite inside an isolated read-only sandbox."""
    if not os.path.exists(db_path):
        return ExecutionResult(
            status="DB_NOT_FOUND", error_message=f"Database file not found: {db_path}"
        )

    if not is_readonly_query(sql):
        return ExecutionResult(
            status="BLOCKED",
            error_message="Query rejected by sandbox safety filter. Only SELECT and WITH statements are allowed.",
        )

    start_time = time.perf_counter()
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        return ExecutionResult(status="CONNECTION_ERROR", error_message=str(e))

    def _timeout_handler() -> int:
        if (time.perf_counter() - start_time) > timeout_seconds:
            return 1
        return 0

    conn.set_progress_handler(_timeout_handler, 1000)

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.perf_counter() - start_time
        return ExecutionResult(status="SUCCESS", result=rows, execution_time=elapsed)
    except sqlite3.OperationalError as e:
        elapsed = time.perf_counter() - start_time
        err_msg = str(e)
        if "interrupted" in err_msg.lower():
            return ExecutionResult(
                status="TIMEOUT",
                error_message="Query execution exceeded deadline.",
                execution_time=elapsed,
            )
        return ExecutionResult(
            status="SYNTAX_ERROR", error_message=err_msg, execution_time=elapsed
        )
    except sqlite3.Error as e:
        elapsed = time.perf_counter() - start_time
        return ExecutionResult(
            status="RUNTIME_ERROR", error_message=str(e), execution_time=elapsed
        )
    finally:
        if conn is not None:
            conn.close()


def probe_physical_literals(
    db_path: str,
    table_column_pairs: list[tuple[str, str]],
    search_term: str,
    limit_per_col: int = 3,
) -> dict[str, list[str]]:
    """Probes physical SQLite tables using LIKE clauses to verify entity literals."""
    if not os.path.exists(db_path) or not search_term.strip():
        return {}

    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    matched_entries: dict[str, list[str]] = {}

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()
        sanitized_term = search_term.replace("%", "").replace("_", "").strip()
        if len(sanitized_term) < 2:
            return {}

        pattern = f"%{sanitized_term}%"
        for table, col in table_column_pairs:
            query = f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" LIKE ? AND "{col}" IS NOT NULL LIMIT ?;'
            try:
                cursor.execute(query, (pattern, limit_per_col))
                matches = [
                    str(row[0]) for row in cursor.fetchall() if row[0] is not None
                ]
                if matches:
                    key = f"{table}.{col}"
                    matched_entries[key] = matches
            except sqlite3.Error:
                continue
    except sqlite3.Error:
        return {}
    finally:
        if conn is not None:
            conn.close()

    return matched_entries
