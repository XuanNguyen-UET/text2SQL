import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DatabaseProfile:
    db_id: str
    tables: list[str]
    columns: dict[str, list[str]]
    column_types: dict[str, dict[str, str]] = field(default_factory=dict)
    primary_keys: dict[str, list[str]] = field(default_factory=dict)
    foreign_keys: list[dict[str, str]] = field(default_factory=list)
    sample_values: dict[str, dict[str, list[Any]]] = field(default_factory=dict)
    column_descriptions: dict[str, dict[str, str]] = field(default_factory=dict)


def preprocess_foreign_keys_accurate(raw_fks: list[Any]) -> list[dict[str, str]]:
    """Standardizes foreign key definitions into the unified schema:

    [{"source": "source_table.source_col", "target": "target_table.target_col"}]
    """
    standardized: list[dict[str, str]] = []
    for item in raw_fks:
        if isinstance(item, dict):
            src = item.get("source") or item.get("from") or ""
            tgt = item.get("target") or item.get("to") or ""
            if src and tgt and "." in src and "." in tgt:
                standardized.append({"source": str(src), "target": str(tgt)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            src, tgt = str(item[0]), str(item[1])
            if "." in src and "." in tgt:
                standardized.append({"source": src, "target": tgt})
    return standardized


def parse_column_descriptions(raw_desc: Any) -> dict[str, dict[str, str]]:
    """Normalizes column documentation mappings across database profiles."""
    if not isinstance(raw_desc, dict):
        return {}
    parsed: dict[str, dict[str, str]] = {}
    for table_name, col_dict in raw_desc.items():
        if isinstance(col_dict, dict):
            parsed[table_name] = {str(k): str(v) for k, v in col_dict.items() if v}
    return parsed


def extract_literal_candidates(text: str) -> list[str]:
    """Extracts quoted strings, capitalized entities, and numerical tokens from NL query."""
    candidates: list[str] = []
    quoted = re.findall(r"['\"`]([^'\"`]+)['\"`]", text)
    candidates.extend(quoted)

    for word in re.findall(r"\b[A-Za-z0-9_\-\.]{2,}\b", text):
        if (
            any(char.isupper() for char in word) or any(char.isdigit() for char in word)
        ) and word not in candidates:
            candidates.append(word)

    return candidates


class DatabaseProfiler:
    """Manages profile loading, structural caching, and literal matching."""

    def __init__(self, profiles_path: str | None = None) -> None:
        self.profiles: dict[str, DatabaseProfile] = {}
        if profiles_path and os.path.exists(profiles_path):
            self.load_profiles(profiles_path)

    def load_profiles(self, profiles_path: str) -> None:
        with open(profiles_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile_dict = (
            data
            if isinstance(data, dict)
            else {item["db_id"]: item for item in data if "db_id" in item}
        )

        for db_id, raw in profile_dict.items():
            fks = preprocess_foreign_keys_accurate(
                raw.get("explicit_foreign_keys", []) or raw.get("foreign_keys", [])
            )
            col_desc = parse_column_descriptions(raw.get("column_descriptions", {}))

            self.profiles[db_id] = DatabaseProfile(
                db_id=db_id,
                tables=list(raw.get("tables", [])),
                columns={k: list(v) for k, v in raw.get("columns", {}).items()},
                column_types=raw.get("column_types", {}),
                primary_keys=raw.get("primary_keys", {}),
                foreign_keys=fks,
                sample_values=raw.get("sample_values", {}),
                column_descriptions=col_desc,
            )

    def get_profile(self, db_id: str) -> DatabaseProfile | None:
        return self.profiles.get(db_id)

    def match_literals_to_profile(
        self,
        profile: DatabaseProfile,
        candidates: list[str],
    ) -> dict[str, list[str]]:
        """Scans in-memory sample values to find matches against candidate literals."""
        matched: dict[str, list[str]] = {}
        for candidate in candidates:
            cand_lower = candidate.lower()
            for tbl, cols in profile.sample_values.items():
                for col, samples in cols.items():
                    for sample in samples:
                        if sample is not None and cand_lower in str(sample).lower():
                            key = f"{tbl}.{col}"
                            matched.setdefault(key, []).append(str(sample))
        return matched
