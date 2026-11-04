import argparse
import json
import logging
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("eval_srr")


def extract_tables_and_columns_from_sql(sql_query: str) -> dict[str, set[str]]:
    """Extracts referenced tables and columns from a SQL string using SQLGlot."""
    tables_to_cols: dict[str, set[str]] = {}
    try:
        parsed = sqlglot.parse_one(sql_query, read="sqlite")
    except (SqlglotError, ValueError) as e:
        logger.warning("Failed to parse SQL query with SQLGlot: %s", e)
        return tables_to_cols

    for table in parsed.find_all(exp.Table):
        tbl_name = table.name
        if tbl_name and tbl_name not in tables_to_cols:
            tables_to_cols[tbl_name] = set()

    for column in parsed.find_all(exp.Column):
        col_name = column.name
        tbl_name = column.table
        if tbl_name and tbl_name in tables_to_cols:
            tables_to_cols[tbl_name].add(col_name)
        elif not tbl_name:
            for cols in tables_to_cols.values():
                cols.add(col_name)

    return tables_to_cols


def compute_srr(
    pruned_schemas: dict[str, dict[str, list[str]]],
    gold_entries: list[dict[str, Any]],
) -> dict[str, float]:
    """Computes Schema Retention Rate (SRR) comparing pruned schemas to Gold SQL elements."""
    table_recalls: list[float] = []
    col_recalls: list[float] = []

    for item in gold_entries:
        qid = str(item.get("question_id", ""))
        gold_sql = item.get("SQL", "") or item.get("query", "")
        if not gold_sql or qid not in pruned_schemas:
            continue

        gold_elements = extract_tables_and_columns_from_sql(gold_sql)
        gold_tables = set(gold_elements.keys())
        pruned_tables = set(pruned_schemas[qid].keys())

        if gold_tables:
            tbl_recall = len(gold_tables & pruned_tables) / len(gold_tables)
            table_recalls.append(tbl_recall)

        all_gold_cols: set[str] = set()
        for tbl, cols in gold_elements.items():
            for c in cols:
                all_gold_cols.add(f"{tbl}.{c}".lower())

        all_pruned_cols: set[str] = set()
        for tbl, cols in pruned_schemas[qid].items():
            for c in cols:
                all_pruned_cols.add(f"{tbl}.{c}".lower())

        if all_gold_cols:
            col_recall = len(all_gold_cols & all_pruned_cols) / len(all_gold_cols)
            col_recalls.append(col_recall)

    avg_tbl = sum(table_recalls) / max(len(table_recalls), 1)
    avg_col = sum(col_recalls) / max(len(col_recalls), 1)
    overall_srr = (avg_tbl + avg_col) / 2.0

    return {
        "overall_srr": round(overall_srr, 4),
        "table_retention_rate": round(avg_tbl, 4),
        "column_retention_rate": round(avg_col, 4),
        "total_evaluated": len(table_recalls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Schema Retention Rate (SRR)")
    parser.add_argument(
        "--gold_file", type=str, required=True, help="Path to dev dataset JSON"
    )
    parser.add_argument(
        "--pruned_file", type=str, required=True, help="Path to pruned schema JSON"
    )
    args = parser.parse_args()

    with open(args.gold_file, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    with open(args.pruned_file, "r", encoding="utf-8") as f:
        pruned_data = json.load(f)

    metrics = compute_srr(pruned_data, gold_data)
    print("=" * 40)
    print("SCHEMA RETENTION RATE (SRR) RESULTS")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
