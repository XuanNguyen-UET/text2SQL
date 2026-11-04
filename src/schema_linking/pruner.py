import logging
import os
import re
from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi

from src.models.engine import HuggingFaceEmbedder
from src.sandbox.executor import probe_physical_literals
from src.schema_linking.profiler import DatabaseProfile, extract_literal_candidates

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ColumnDoc:
    table: str
    column: str
    text: str


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridSchemaPruner:
    """Prunes schema tables and columns using BM25, dense BGE similarity, and Steiner Trees."""

    def __init__(
        self,
        embedder: HuggingFaceEmbedder,
        bm25_alpha: float = 0.6,
        semantic_beta: float = 0.4,
        threshold_ratio: float = 0.15,
        max_tables_final: int = 6,
        max_columns_per_table: int = 10,
    ) -> None:
        self.embedder = embedder
        self.bm25_alpha = bm25_alpha
        self.semantic_beta = semantic_beta
        self.threshold_ratio = threshold_ratio
        self.max_tables_final = max_tables_final
        self.max_columns_per_table = max_columns_per_table

    def _build_column_documents(self, profile: DatabaseProfile) -> list[ColumnDoc]:
        docs: list[ColumnDoc] = []
        for table, cols in profile.columns.items():
            for col in cols:
                desc = profile.column_descriptions.get(table, {}).get(col, "")
                samples = profile.sample_values.get(table, {}).get(col, [])
                samples_str = " ".join(str(s) for s in samples[:3])
                text = f"Table: {table} Column: {col} Description: {desc} Samples: {samples_str}"
                docs.append(ColumnDoc(table=table, column=col, text=text))
        return docs

    def _build_table_graph(self, profile: DatabaseProfile) -> nx.Graph:
        graph = nx.Graph()
        for tbl in profile.tables:
            graph.add_node(tbl)
        for fk in profile.foreign_keys:
            src_tbl = fk["source"].split(".")[0]
            tgt_tbl = fk["target"].split(".")[0]
            if src_tbl in profile.tables and tgt_tbl in profile.tables:
                graph.add_edge(src_tbl, tgt_tbl)
        return graph

    def prune(
        self,
        profile: DatabaseProfile,
        question: str,
        evidence: str = "",
        db_path: str | None = None,
    ) -> dict[str, list[str]]:
        col_docs = self._build_column_documents(profile)
        if not col_docs:
            return {
                tbl: cols[: self.max_columns_per_table]
                for tbl, cols in profile.columns.items()
            }

        corpus = [tokenize(d.text) for d in col_docs]
        bm25 = BM25Okapi(corpus)

        query_text = f"{question} {evidence}".strip()
        query_tokens = tokenize(query_text)
        bm25_scores = np.array(bm25.get_scores(query_tokens), dtype=np.float32)
        if np.max(bm25_scores) > 0:
            bm25_scores /= np.max(bm25_scores)

        doc_texts = [d.text for d in col_docs]
        doc_embeddings = self.embedder.encode(doc_texts)
        query_embedding = self.embedder.encode([query_text])[0]
        semantic_scores = np.dot(doc_embeddings, query_embedding)
        semantic_scores = (semantic_scores + 1.0) / 2.0

        hybrid_scores = (self.bm25_alpha * bm25_scores) + (
            self.semantic_beta * semantic_scores
        )
        max_score = float(np.max(hybrid_scores)) if len(hybrid_scores) > 0 else 0.0
        cutoff = max_score * self.threshold_ratio

        selected_table_columns: dict[str, set[str]] = {}
        for doc, score in zip(col_docs, hybrid_scores):
            if score >= cutoff:
                selected_table_columns.setdefault(doc.table, set()).add(doc.column)

        if db_path and os.path.exists(db_path):
            candidates = extract_literal_candidates(query_text)
            pairs = [(d.table, d.column) for d in col_docs]
            for cand in candidates[:3]:
                probed = probe_physical_literals(db_path, pairs, cand, limit_per_col=1)
                for matched_key in probed:
                    tbl, col = matched_key.split(".")
                    selected_table_columns.setdefault(tbl, set()).add(col)

        table_graph = self._build_table_graph(profile)
        terminal_nodes = [t for t in selected_table_columns if t in table_graph]

        connected_tables: set[str] = set(terminal_nodes)
        if len(terminal_nodes) > 1:
            try:
                components = list(nx.connected_components(table_graph))
                if components:
                    sub_graph = table_graph.subgraph(max(components, key=len))
                    valid_terminals = [t for t in terminal_nodes if t in sub_graph]
                    if len(valid_terminals) > 1:
                        st_tree = nx.approximation.steiner_tree(
                            sub_graph, valid_terminals
                        )
                        connected_tables.update(st_tree.nodes())
            except (nx.NetworkXError, nx.NetworkXException, ValueError) as exc:
                logger.warning("Steiner tree computation fallback: %s", exc)

        final_tables = list(connected_tables)[: self.max_tables_final]
        if not final_tables and profile.tables:
            final_tables = profile.tables[:2]

        result: dict[str, list[str]] = {}
        for tbl in final_tables:
            cols = list(selected_table_columns.get(tbl, set()))
            all_tbl_cols = profile.columns.get(tbl, [])
            for pk in profile.primary_keys.get(tbl, []):
                if pk not in cols and pk in all_tbl_cols:
                    cols.append(pk)
            for fk in profile.foreign_keys:
                src_tbl, src_col = fk["source"].split(".")
                tgt_tbl, tgt_col = fk["target"].split(".")
                if src_tbl == tbl and src_col in all_tbl_cols and src_col not in cols:
                    cols.append(src_col)
                if tgt_tbl == tbl and tgt_col in all_tbl_cols and tgt_col not in cols:
                    cols.append(tgt_col)

            if not cols:
                cols = all_tbl_cols[: self.max_columns_per_table]
            result[tbl] = cols[: self.max_columns_per_table]

        return result


class FastSchemaPruner:
    """Heuristic rule-boosted BM25 pruner with BFS shortest path connectivity."""

    def __init__(
        self, max_tables_final: int = 6, max_columns_per_table: int = 10
    ) -> None:
        self.max_tables_final = max_tables_final
        self.max_columns_per_table = max_columns_per_table

    def prune(
        self,
        profile: DatabaseProfile,
        question: str,
        evidence: str = "",
    ) -> dict[str, list[str]]:
        query_text = f"{question} {evidence}".lower()
        query_tokens = set(tokenize(query_text))

        table_scores: dict[str, float] = {t: 0.0 for t in profile.tables}
        column_hits: dict[str, set[str]] = {t: set() for t in profile.tables}

        for tbl in profile.tables:
            if tbl.lower() in query_tokens:
                table_scores[tbl] += 2.0

            for col in profile.columns.get(tbl, []):
                col_lower = col.lower()
                if f"`{col_lower}`" in query_text:
                    table_scores[tbl] += 8.0
                    column_hits[tbl].add(col)
                elif col_lower in query_tokens:
                    table_scores[tbl] += 4.0
                    column_hits[tbl].add(col)

                samples = profile.sample_values.get(tbl, {}).get(col, [])
                for sample in samples:
                    if sample is not None and str(sample).lower() in query_text:
                        table_scores[tbl] += 10.0
                        column_hits[tbl].add(col)

        ranked_tables = sorted(
            [t for t, score in table_scores.items() if score > 0],
            key=lambda t: table_scores[t],
            reverse=True,
        )

        selected_tables = set(ranked_tables[: self.max_tables_final])
        if not selected_tables and profile.tables:
            selected_tables.add(profile.tables[0])

        adjacency: dict[str, set[str]] = {t: set() for t in profile.tables}
        for fk in profile.foreign_keys:
            src_tbl = fk["source"].split(".")[0]
            tgt_tbl = fk["target"].split(".")[0]
            if src_tbl in adjacency and tgt_tbl in adjacency:
                adjacency[src_tbl].add(tgt_tbl)
                adjacency[tgt_tbl].add(src_tbl)

        if len(selected_tables) > 1:
            table_list = list(selected_tables)
            start = table_list[0]
            for target in table_list[1:]:
                queue: deque[tuple[str, list[str]]] = deque([(start, [start])])
                visited = {start}
                while queue:
                    curr, path = queue.popleft()
                    if curr == target:
                        selected_tables.update(path)
                        break
                    for neighbor in adjacency.get(curr, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))

        result: dict[str, list[str]] = {}
        for tbl in list(selected_tables)[: self.max_tables_final]:
            cols = list(column_hits.get(tbl, set()))
            all_cols = profile.columns.get(tbl, [])
            for pk in profile.primary_keys.get(tbl, []):
                if pk in all_cols and pk not in cols:
                    cols.append(pk)
            if not cols:
                cols = all_cols[: self.max_columns_per_table]
            result[tbl] = cols[: self.max_columns_per_table]

        return result


def prune_schema(
    mode: str,
    profile: DatabaseProfile,
    question: str,
    evidence: str = "",
    embedder: HuggingFaceEmbedder | None = None,
    db_path: str | None = None,
) -> dict[str, list[str]]:
    """Unified entrypoint routing schema pruning to hybrid or fast engines."""
    if mode == "hybrid" and embedder is not None:
        pruner = HybridSchemaPruner(embedder=embedder)
        return pruner.prune(profile, question, evidence, db_path=db_path)
    return FastSchemaPruner().prune(profile, question, evidence)
