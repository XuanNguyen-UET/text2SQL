import re
from typing import Any

from src.models.engine import HFQuantizedRunner, VLLMRunner, apply_chat_template_safe
from src.models.prompts import MINI_DDL_SYSTEM_PROMPT, MINI_DDL_USER_TEMPLATE
from src.schema_linking.profiler import DatabaseProfile


def render_algorithmic_ddl(
    profile: DatabaseProfile,
    pruned_schema: dict[str, list[str]],
    sample_limit: int = 3,
) -> str:
    """Renders deterministic SQLite CREATE TABLE statements with inline samples and constraints."""
    ddl_statements: list[str] = []

    for table, cols in pruned_schema.items():
        lines: list[str] = []
        for col in cols:
            col_type = profile.column_types.get(table, {}).get(col, "TEXT")
            samples = profile.sample_values.get(table, {}).get(col, [])[:sample_limit]
            desc = profile.column_descriptions.get(table, {}).get(col, "")

            comment_parts: list[str] = []
            if samples:
                comment_parts.append(f"samples: {samples}")
            if desc:
                comment_parts.append(f"note: {desc}")

            comment_str = f" -- {'; '.join(comment_parts)}" if comment_parts else ""
            lines.append(f'  "{col}" {col_type}{comment_str}')

        pks = [pk for pk in profile.primary_keys.get(table, []) if pk in cols]
        if pks:
            pk_str = ", ".join(f'"{pk}"' for pk in pks)
            lines.append(f"  PRIMARY KEY ({pk_str})")

        for fk in profile.foreign_keys:
            src_tbl, src_col = fk["source"].split(".")
            tgt_tbl, tgt_col = fk["target"].split(".")
            if src_tbl == table and src_col in cols and tgt_tbl in pruned_schema:
                lines.append(f'  FOREIGN KEY ("{src_col}") REFERENCES "{tgt_tbl}" ("{tgt_col}")')

        table_def = f'CREATE TABLE "{table}" (\n' + ",\n".join(lines) + "\n);"
        ddl_statements.append(table_def)

    return "\n\n".join(ddl_statements)


def validate_mini_ddl_against_profile(mini_ddl: str, profile: DatabaseProfile) -> bool:
    """Validates that all tables and columns declared in mini-DDL exist in the database profile."""
    table_pattern = re.compile(r'CREATE\s+TABLE\s+["`]?(\w+)["`]?', re.IGNORECASE)
    declared_tables = table_pattern.findall(mini_ddl)
    if not declared_tables:
        return False

    for tbl in declared_tables:
        if tbl not in profile.tables:
            return False

    col_pattern = re.compile(r'^\s*["`]?(\w+)["`]?\s+(?:TEXT|INTEGER|REAL|BLOB|NUMERIC|VARCHAR)', re.IGNORECASE | re.MULTILINE)
    declared_cols = col_pattern.findall(mini_ddl)
    all_valid_cols = {col for col_list in profile.columns.values() for col in col_list}

    for col in declared_cols:
        if col.upper() in {"PRIMARY", "FOREIGN", "KEY", "CONSTRAINT", "UNIQUE"}:
            continue
        if col not in all_valid_cols:
            return False

    return True


def generate_mini_ddl(
    runner: VLLMRunner | HFQuantizedRunner,
    profile: DatabaseProfile,
    pruned_schema: dict[str, list[str]],
    question: str,
    evidence: str = "",
    logical_plan: str = "",
) -> str:
    """Generates and validates compressed Mini-DDL, falling back to algorithmic DDL upon failure."""
    fallback_ddl = render_algorithmic_ddl(profile, pruned_schema)

    profile_text = fallback_ddl
    user_content = MINI_DDL_USER_TEMPLATE.format(
        db_profile_text=profile_text,
        question=question,
        evidence=evidence or "None",
        logical_plan=logical_plan or "None",
    )

    messages = [
        {"role": "system", "content": MINI_DDL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    prompt = apply_chat_template_safe(runner.tokenizer, messages, add_generation_prompt=True)

    try:
        if isinstance(runner, VLLMRunner):
            outputs = runner.generate([prompt], max_tokens=2048, temperature=0.0)
            raw_output = outputs[0][0]
        else:
            raw_output = runner.generate(prompt, max_new_tokens=2048, temperature=0.0)

        raw_output = re.sub(r"```sql|```", "", raw_output).strip()
        if validate_mini_ddl_against_profile(raw_output, profile):
            return raw_output
    except Exception:
        pass

    return fallback_ddl
