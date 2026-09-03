"""Unit tests for LIBERO-Plus category adapter (no MuJoCo required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from libero_plus_eval_utils import (
    LIBERO_PLUS_SUITES,
    SuiteView,
    build_suite,
    resolve_category_values,
    sanitize_category_value,
)


class _FakeSuite:
    def __init__(self, name: str, task_names: list[str]):
        self.name = name
        self._names = list(task_names)
        self.n_tasks = len(self._names)
        self._tasks = [f"task:{n}" for n in self._names]
        self._inits = [[i] for i in range(self.n_tasks)]

    def get_task_names(self) -> list[str]:
        return list(self._names)

    def get_task(self, i: int):
        return self._tasks[i]

    def get_task_init_states(self, i: int):
        return self._inits[i]


def _fake_benchmark_dict() -> dict:
    return {
        "libero_spatial": lambda: _FakeSuite(
            "libero_spatial",
            ["spatial_a_bg", "spatial_b_cam", "spatial_c_bg"],
        ),
        "libero_object": lambda: _FakeSuite(
            "libero_object",
            ["object_a_bg", "object_b_light"],
        ),
        "libero_goal": lambda: _FakeSuite("libero_goal", ["goal_a_bg"]),
        "libero_10": lambda: _FakeSuite("libero_10", ["ten_a_bg", "ten_b_cam"]),
    }


def _classification() -> dict:
    return {
        "libero_spatial": [
            {"name": "spatial_a_bg", "category": "Background Textures"},
            {"name": "spatial_b_cam", "category": "Camera Viewpoints"},
            {"name": "spatial_c_bg", "category": "Background Textures"},
        ],
        "libero_object": [
            {"name": "object_a_bg", "category": "Background Textures"},
            {"name": "object_b_light", "category": "Light Conditions"},
        ],
        "libero_goal": [
            {"name": "goal_a_bg", "category": "Background Textures"},
        ],
        "libero_10": [
            {"name": "ten_a_bg", "category": "Background Textures"},
            {"name": "ten_b_cam", "category": "Camera Viewpoints"},
        ],
    }


def test_sanitize_category_value_spaces():
    assert sanitize_category_value("Background Textures") == "Background_Textures"


def test_resolve_category_values_passthrough():
    assert resolve_category_values(["Camera Viewpoints", " Light Conditions "]) == [
        "Camera Viewpoints",
        "Light Conditions",
    ]


def test_build_suite_no_category_keeps_all():
    view = build_suite(
        "libero_spatial",
        None,
        benchmark_dict=_fake_benchmark_dict(),
        classification=_classification(),
    )
    assert view.n_tasks == 3
    assert view.source_task_id(0) == 0
    assert view.get_task(1) == "task:spatial_b_cam"


def test_build_suite_category_filter_stable_ids():
    view = build_suite(
        "libero_spatial",
        "Background Textures",
        benchmark_dict=_fake_benchmark_dict(),
        classification=_classification(),
    )
    assert view.n_tasks == 2
    assert view.source_task_id(0) == 0
    assert view.source_task_id(1) == 2
    assert view.get_task_names() == ["spatial_a_bg", "spatial_c_bg"]
    assert view.get_task_init_states(1) == [2]


def test_build_suite_unknown_raises():
    with pytest.raises(ValueError, match="Unknown suite"):
        build_suite("libero_nope", None, benchmark_dict=_fake_benchmark_dict())


def test_libero_mix_aggregate_by_category():
    view = build_suite(
        "libero_mix",
        "Background Textures",
        benchmark_dict=_fake_benchmark_dict(),
        classification=_classification(),
    )
    # spatial(2) + object(1) + goal(1) + libero_10(1) = 5
    assert view.n_tasks == 5
    assert view.name == "libero_mix"
    assert view.get_task(0) == "task:spatial_a_bg"
    assert view.get_task(2) == "task:object_a_bg"


def test_libero_mix_no_category_all_tasks():
    view = build_suite(
        "libero_mix",
        None,
        benchmark_dict=_fake_benchmark_dict(),
        classification=_classification(),
    )
    assert view.n_tasks == 3 + 2 + 1 + 2
