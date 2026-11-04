import logging
import re

logger = logging.getLogger(__name__)

SQL_ANSWER_REGEX = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE
)
THINK_REGEX = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def validate_sft_record(record: dict[str, str]) -> tuple[bool, str]:
    """Validates that a single SFT training record adheres to required reasoning tags and SQL syntax."""
    output_text = record.get("output", "").strip()
    if not output_text:
        return False, "Empty output field."

    if not THINK_REGEX.search(output_text):
        return False, "Missing <think>...</think> reasoning block."

    answer_match = SQL_ANSWER_REGEX.search(output_text)
    if not answer_match:
        return False, "Missing <answer>...</answer> target block."

    sql = answer_match.group(1).strip()
    if not sql:
        return False, "Empty SQL query inside <answer> tag."

    normalized_sql = sql.upper()
    if not normalized_sql.startswith(("SELECT", "WITH")):
        return False, f"SQL query does not start with SELECT or WITH: {sql[:30]}"

    return True, "Valid"


def validate_sft_records(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Filters dataset records into valid and rejected subsets with detailed logging."""
    valid_records: list[dict[str, str]] = []
    rejected_records: list[dict[str, str]] = []

    for idx, rec in enumerate(records):
        is_valid, reason = validate_sft_record(rec)
        if is_valid:
            valid_records.append(rec)
        else:
            logger.warning("Record index %d rejected: %s", idx, reason)
            rejected_records.append(
                {"index": str(idx), "reason": reason, "record": str(rec)}
            )

    return valid_records, rejected_records


def clean_input_seq_for_sft(
    mini_ddl: str,
    question: str,
    evidence: str = "",
) -> list[dict[str, str]]:
    """Constructs standard chat messages for SFT training input sequences."""
    user_prompt = (
        f"Database Schema (DDL):\n{mini_ddl}\n\n"
        f"Question: {question}\n"
        f"Evidence: {evidence or 'None'}\n\n"
        "Provide your detailed reasoning within <think>...</think> and place the executable "
        "SQLite query strictly within <answer>...</answer>."
    )
    return [
        {
            "role": "system",
            "content": "You are an elite SQL Engineer specializing in SQLite dialect.",
        },
        {"role": "user", "content": user_prompt},
    ]
