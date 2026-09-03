"""Helpers for LIBERO-Plus multi-suite / multi-category evaluation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LIBERO_PLUS_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

_DEFAULT_LIBERO_PKG_ROOT = Path("/path/to/LIBERO-plus")
LIBERO_PKG_ROOT = Path(os.environ.get("LIBERO_PKG_ROOT", _DEFAULT_LIBERO_PKG_ROOT))

TASK_CLASSIFICATION_PATH = (
    LIBERO_PKG_ROOT
    / "libero"
    / "libero"
    / "benchmark"
    / "task_classification.json"
)


def sanitize_category_value(category_value: str) -> str:
    return str(category_value).strip().replace(" ", "_")


def get_all_category_values(
    classification_path: Path | None = None,
) -> list[str]:
    path = classification_path or TASK_CLASSIFICATION_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories: set[str] = set()
    for tasks in data.values():
        for task in tasks:
            cat = task.get("category")
            if cat:
                categories.add(str(cat))
    return sorted(categories)


def resolve_category_values(
    category_values: Iterable[str] | None,
) -> list[str]:
    if category_values is None:
        return get_all_category_values()
    resolved = [str(v).strip() for v in category_values if str(v).strip()]
    if not resolved:
        return get_all_category_values()
    return resolved


def load_task_classification(
    classification_path: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    path = classification_path or TASK_CLASSIFICATION_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"task_classification.json must be an object: {path}")
    return data


@dataclass
class SuiteView:
    """Filtered / aggregated view over upstream suite object(s)."""

    name: str
    category_value: str | None
    _entries: list[tuple[Any, int]]  # (suite_obj, source_task_id)

    @property
    def n_tasks(self) -> int:
        return len(self._entries)

    def source_task_id(self, i: int) -> int:
        return int(self._entries[i][1])

    def get_task(self, i: int) -> Any:
        suite_obj, src = self._entries[i]
        return suite_obj.get_task(src)

    def get_task_init_states(self, i: int) -> Any:
        suite_obj, src = self._entries[i]
        return suite_obj.get_task_init_states(src)

    def get_task_names(self) -> list[str]:
        names: list[str] = []
        for suite_obj, src in self._entries:
            all_names = suite_obj.get_task_names()
            names.append(all_names[src])
        return names


def _instantiate_upstream_suite(suite_name: str, benchmark_dict: dict[str, Any] | None = None) -> Any:
    if suite_name == "libero_mix":
        raise ValueError(
            "libero_mix must be built via build_suite(); upstream LIBERO_MIX is unsupported"
        )
    if suite_name not in LIBERO_PLUS_SUITES:
        raise ValueError(f"Unknown suite: {suite_name!r}. Expected one of {LIBERO_PLUS_SUITES}")
    if benchmark_dict is None:
        from libero.libero import benchmark

        benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise KeyError(f"Suite {suite_name!r} missing from benchmark_dict")
    # Upstream ctors only accept task_order_index — never pass category_value.
    return benchmark_dict[suite_name]()


def _name_to_source_index(suite_obj: Any) -> dict[str, int]:
    names = list(suite_obj.get_task_names())
    mapping: dict[str, int] = {}
    for idx, name in enumerate(names):
        mapping[str(name)] = idx
    return mapping


def _filter_source_indices(
    suite_name: str,
    suite_obj: Any,
    category_value: str | None,
    classification: dict[str, list[dict[str, Any]]] | None,
) -> list[int]:
    n = int(suite_obj.n_tasks)
    if category_value is None or str(category_value).strip() == "":
        return list(range(n))

    if classification is None:
        classification = load_task_classification()
    entries = classification.get(suite_name)
    if entries is None:
        raise KeyError(
            f"Suite {suite_name!r} missing from task_classification.json "
            f"(keys={sorted(classification.keys())})"
        )

    name_to_idx = _name_to_source_index(suite_obj)
    cat = str(category_value).strip()
    source_ids: list[int] = []
    missing = 0
    for entry in entries:
        if str(entry.get("category", "")).strip() != cat:
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        src = name_to_idx.get(name)
        if src is None:
            missing += 1
            continue
        source_ids.append(src)

    if not source_ids:
        raise ValueError(
            f"No tasks for suite={suite_name!r} category={cat!r} "
            f"(missing_name_matches={missing})"
        )
    # Stable order: ascending source index, unique
    return sorted(set(source_ids))


def build_suite(
    suite_name: str,
    category_value: str | None = None,
    *,
    benchmark_dict: dict[str, Any] | None = None,
    classification: dict[str, list[dict[str, Any]]] | None = None,
) -> SuiteView:
    """Build a SuiteView compatible with eval workers.

    - Base suites: instantiate without category_value; optionally filter by classification.
    - libero_mix: deterministic aggregate of the four base suites (same category filter).
    """
    cat = None if category_value is None or str(category_value).strip() == "" else str(category_value).strip()

    if suite_name == "libero_mix":
        entries: list[tuple[Any, int]] = []
        for base_name in LIBERO_PLUS_SUITES:
            base = _instantiate_upstream_suite(base_name, benchmark_dict=benchmark_dict)
            for src in _filter_source_indices(base_name, base, cat, classification):
                entries.append((base, src))
        if not entries:
            raise ValueError(f"libero_mix empty for category={cat!r}")
        return SuiteView(name="libero_mix", category_value=cat, _entries=entries)

    suite_obj = _instantiate_upstream_suite(suite_name, benchmark_dict=benchmark_dict)
    source_ids = _filter_source_indices(suite_name, suite_obj, cat, classification)
    return SuiteView(
        name=suite_name,
        category_value=cat,
        _entries=[(suite_obj, src) for src in source_ids],
    )
