import os
import sqlite3
import tempfile

import pytest

from src.agent.orchestrator import extract_sql_from_response
from src.sandbox.executor import execute_sql, is_readonly_query
from src.schema_linking.ddl_rewriter import validate_mini_ddl_against_profile
from src.schema_linking.profiler import DatabaseProfile
from src.training.rewards import (
    execution_reward_func,
    format_reward_func,
)


@pytest.fixture
def temp_test_db() -> str:
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.sqlite")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
    )
    cursor.execute("INSERT INTO users VALUES (1, 'Alice', 25), (2, 'Bob', 30);")
    conn.commit()
    conn.close()
    return db_path


def test_sandbox_allows_select(temp_test_db: str) -> None:
    res = execute_sql(temp_test_db, "SELECT name FROM users WHERE id = 1;")
    assert res.status == "SUCCESS"
    assert res.result == [("Alice",)]


def test_sandbox_blocks_dml(temp_test_db: str) -> None:
    assert not is_readonly_query("DROP TABLE users;")
    assert not is_readonly_query("INSERT INTO users VALUES (3, 'Charlie', 35);")
    assert not is_readonly_query("DELETE FROM users WHERE id = 1;")

    blocked_res = execute_sql(temp_test_db, "DROP TABLE users;")
    assert blocked_res.status == "BLOCKED"


def test_sql_extraction_from_answer_tags() -> None:
    sample_text = (
        "<think>Let's find the max age from users table.</think>\n"
        "<answer>\nSELECT MAX(age) FROM users;\n</answer>"
    )
    sql, is_valid = extract_sql_from_response(sample_text)
    assert is_valid is True
    assert sql == "SELECT MAX(age) FROM users;"


def test_sql_extraction_markdown_fallback() -> None:
    sample_text = (
        "Here is the query you requested:\n```sql\nSELECT name FROM users;\n```"
    )
    sql, is_valid = extract_sql_from_response(sample_text)
    assert is_valid is False
    assert sql == "SELECT name FROM users;"


def test_format_reward_evaluation() -> None:
    valid_completion = [
        [{"content": "<think>Thinking</think><answer>SELECT 1;</answer>"}]
    ]
    partial_completion = [[{"content": "<answer>SELECT 1;</answer>"}]]
    invalid_completion = [[{"content": "Just a query SELECT 1;"}]]

    assert format_reward_func([], valid_completion)[0] == 1.0
    assert format_reward_func([], partial_completion)[0] == 0.5
    assert format_reward_func([], invalid_completion)[0] == 0.0


def test_execution_reward(temp_test_db: str) -> None:
    good_completion = [[{"content": "<answer>SELECT * FROM users;</answer>"}]]
    bad_completion = [
        [{"content": "<answer>SELECT * FROM non_existing_table;</answer>"}]
    ]

    good_reward = execution_reward_func([], good_completion, db_path=temp_test_db)
    bad_reward = execution_reward_func([], bad_completion, db_path=temp_test_db)

    assert good_reward[0] == 1.0
    assert bad_reward[0] == -0.5


def test_mini_ddl_validation() -> None:
    profile = DatabaseProfile(
        db_id="test_db",
        tables=["users", "orders"],
        columns={"users": ["id", "name"], "orders": ["order_id", "user_id"]},
    )

    valid_ddl = 'CREATE TABLE "users" (\n  "id" INTEGER,\n  "name" TEXT\n);'
    assert validate_mini_ddl_against_profile(valid_ddl, profile) is True

    hallucinated_table_ddl = 'CREATE TABLE "accounts" (\n  "id" INTEGER\n);'
    assert validate_mini_ddl_against_profile(hallucinated_table_ddl, profile) is False

    hallucinated_col_ddl = (
        'CREATE TABLE "users" (\n  "id" INTEGER,\n  "salary" REAL\n);'
    )
    assert validate_mini_ddl_against_profile(hallucinated_col_ddl, profile) is False
