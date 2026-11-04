# src/sandbox/summarizer.py
from collections import defaultdict
from typing import Any

from src.sandbox.executor import ExecutionResult


def canonicalize_rows(
    rows: list[tuple[Any, ...]] | None,
) -> tuple[tuple[str, ...], ...] | None:
    """Canonicalizes result rows into an order-invariant hashable tuple for consensus comparison."""
    if rows is None:
        return None
    normalized_list: list[tuple[str, ...]] = []
    for row in rows:
        normalized_row = tuple(str(val) if val is not None else "NULL" for val in row)
        normalized_list.append(normalized_row)
    return tuple(sorted(normalized_list))


def summarize_execution_candidates(
    candidate_sqls: list[str],
    execution_results: list[ExecutionResult],
) -> dict[str, Any]:
    """Aggregates execution results across candidates into consensus clusters and actionable summaries."""
    clusters: dict[tuple[tuple[str, ...], ...] | None, list[int]] = defaultdict(list)
    successful_indices: list[int] = []
    error_traces: dict[int, str] = {}

    for idx, (sql, exec_res) in enumerate(zip(candidate_sqls, execution_results)):
        if exec_res.status == "SUCCESS":
            canonical_key = canonicalize_rows(exec_res.result)
            clusters[canonical_key].append(idx)
            successful_indices.append(idx)
        else:
            error_traces[idx] = f"[{exec_res.status}] {exec_res.error_message}"

    dominant_cluster_indices: list[int] = []
    if successful_indices:
        for key, members in clusters.items():
            if key is not None and len(members) > len(dominant_cluster_indices):
                dominant_cluster_indices = members

    consensus_reached = len(dominant_cluster_indices) >= 2
    recommended_sql = ""
    if dominant_cluster_indices:
        recommended_sql = min(
            (candidate_sqls[i] for i in dominant_cluster_indices),
            key=len,
        )
    elif successful_indices:
        recommended_sql = min(
            (candidate_sqls[i] for i in successful_indices),
            key=len,
        )

    return {
        "total_candidates": len(candidate_sqls),
        "success_count": len(successful_indices),
        "consensus_reached": consensus_reached,
        "dominant_cluster_size": len(dominant_cluster_indices),
        "dominant_indices": dominant_cluster_indices,
        "recommended_sql": recommended_sql,
        "errors": error_traces,
    }
