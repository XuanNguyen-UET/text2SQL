import logging
import re

from src.sandbox.executor import execute_sql

logger = logging.getLogger(__name__)

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_sql_from_completion(text: str) -> str:
    """Extracts SQL query string from model completion."""
    match = ANSWER_PATTERN.search(text)
    if match:
        sql = match.group(1).strip()
        return re.sub(r"^```sql|^```|```$", "", sql, flags=re.MULTILINE).strip()
    return ""


def format_reward_func(
    prompts: list[str], completions: list[list[dict[str, str]]], **kwargs: object
) -> list[float]:
    """Rewards strict adherence to <think>...</think><answer>...</answer> XML formatting."""
    rewards: list[float] = []
    for completion in completions:
        content = completion[0]["content"] if completion else ""
        has_think = bool(THINK_PATTERN.search(content))
        has_answer = bool(ANSWER_PATTERN.search(content))

        if has_think and has_answer:
            rewards.append(1.0)
        elif has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def execution_reward_func(
    prompts: list[str],
    completions: list[list[dict[str, str]]],
    db_paths: list[str] | None = None,
    **kwargs: object,
) -> list[float]:
    """Evaluates SQL execution status inside the SQLite Sandbox."""
    rewards: list[float] = []
    default_db = kwargs.get("db_path", "")

    for idx, completion in enumerate(completions):
        content = completion[0]["content"] if completion else ""
        sql = extract_sql_from_completion(content)
        if not sql:
            rewards.append(-1.0)
            continue

        target_db = (
            db_paths[idx] if (db_paths and idx < len(db_paths)) else str(default_db)
        )
        if not target_db:
            rewards.append(0.0)
            continue

        res = execute_sql(target_db, sql, timeout_seconds=5.0)
        if res.status == "SUCCESS":
            rewards.append(1.0 if res.result else 0.5)
        elif res.status in ("SYNTAX_ERROR", "RUNTIME_ERROR"):
            rewards.append(-0.5)
        elif res.status in ("TIMEOUT", "BLOCKED"):
            rewards.append(-1.0)
        else:
            rewards.append(-0.5)

    return rewards


def match_reward_func(
    prompts: list[str],
    completions: list[list[dict[str, str]]],
    gold_sqls: list[str] | None = None,
    db_paths: list[str] | None = None,
    **kwargs: object,
) -> list[float]:
    """Rewards execution equivalence between generated candidate and Gold SQL."""
    rewards: list[float] = []
    if not gold_sqls:
        return [0.0] * len(completions)

    default_db = kwargs.get("db_path", "")

    for idx, completion in enumerate(completions):
        content = completion[0]["content"] if completion else ""
        cand_sql = extract_sql_from_completion(content)
        gold_sql = gold_sqls[idx] if idx < len(gold_sqls) else ""

        if not cand_sql or not gold_sql:
            rewards.append(0.0)
            continue

        target_db = (
            db_paths[idx] if (db_paths and idx < len(db_paths)) else str(default_db)
        )
        if not target_db:
            rewards.append(0.0)
            continue

        cand_res = execute_sql(target_db, cand_sql, timeout_seconds=5.0)
        gold_res = execute_sql(target_db, gold_sql, timeout_seconds=5.0)

        if cand_res.status == "SUCCESS" and gold_res.status == "SUCCESS":
            cand_rows = set(cand_res.result or [])
            gold_rows = set(gold_res.result or [])
            if cand_rows == gold_rows and cand_rows:
                rewards.append(2.0)
            elif cand_rows == gold_rows:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards
