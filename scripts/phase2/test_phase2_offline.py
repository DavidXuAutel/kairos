"""Offline Phase-2 safety skeleton tests (no robot)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action_pipeline import denormalize_actions
from arming import ArmingGate
from fake_controller import FakeController
from limits import clamp_eef_delta, reject_if_outside_workspace
from offline_runner import replay_actions_jsonl
from state_builder import (
    build_proprio,
    build_proprio_from_franka,
    map_franka_gripper_to_libero,
    quat2axisangle,
)


def test_build_proprio_shape():
    p = build_proprio(
        np.zeros(3),
        np.array([0.1, 0.0, 0.0]),
        np.array([0.04, -0.04]),
    )
    assert p.shape == (8,)
    assert p.dtype == np.float32


def test_build_proprio_rejects_bad_shape():
    with pytest.raises(ValueError):
        build_proprio(np.zeros(2), np.zeros(3), np.zeros(2))


def test_build_proprio_rejects_nan():
    with pytest.raises(ValueError):
        build_proprio(np.array([np.nan, 0, 0]), np.zeros(3), np.zeros(2))


def test_gripper_antisym():
    g = map_franka_gripper_to_libero(np.array([0.04, 0.04]))
    np.testing.assert_allclose(g, [0.04, -0.04], atol=1e-6)
    g2 = map_franka_gripper_to_libero(np.array([0.0, 0.0]))
    np.testing.assert_allclose(g2, [0.0, 0.0], atol=1e-6)


def test_origin_and_libero_aa():
    # identity quat + identity R_offset → zero aa
    p = build_proprio_from_franka(
        np.array([0.55, 0.1, 0.3]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([0.04, 0.04]),
        origin_xyz=(0.45, 0.0, 0.0),
        r_offset=np.eye(3),
    )
    np.testing.assert_allclose(p[:3], [0.10, 0.10, 0.30], atol=1e-5)
    np.testing.assert_allclose(p[3:6], [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(p[6:], [0.04, -0.04], atol=1e-5)
    aa = quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(aa, [0, 0, 0], atol=1e-6)


def test_libero_gripper_map():
    from gripper_map import libero_gripper_to_percent, libero_gripper_wants_close

    assert abs(libero_gripper_to_percent(-1.0) - 1.0) < 1e-6
    assert abs(libero_gripper_to_percent(1.0) - 0.0) < 1e-6
    assert libero_gripper_wants_close(0.2)
    assert not libero_gripper_wants_close(-0.2)


def test_approach_align_requires_xy():
    from approach_align import approach_is_aligned

    # near_table alone must NOT align
    ok, info = approach_is_aligned(
        limited_xyz_inf=0.04,
        limited_xy_inf=0.04,
        ee_z=0.20,
        grip_denorm=0.9,
        approach_converge_xyz=0.012,
        approach_converge_xy=0.015,
        grasp_z=0.18,
        align_z_margin=0.03,
    )
    assert not ok
    assert info["near_table"] and info["close_intent"] and not info["xy_ok"]

    # xy small + close near table → align
    ok2, info2 = approach_is_aligned(
        limited_xyz_inf=0.04,
        limited_xy_inf=0.01,
        ee_z=0.20,
        grip_denorm=0.9,
        approach_converge_xyz=0.012,
        approach_converge_xy=0.015,
        grasp_z=0.18,
        align_z_margin=0.03,
    )
    assert ok2 and info2["xy_ok"]

    # full xyz converge + xy ok → align even mid-air
    ok3, _ = approach_is_aligned(
        limited_xyz_inf=0.01,
        limited_xy_inf=0.01,
        ee_z=0.40,
        grip_denorm=-0.9,
        approach_converge_xyz=0.012,
        approach_converge_xy=0.015,
        grasp_z=0.18,
        align_z_margin=0.03,
    )
    assert ok3


def test_descend_tracking_abort_and_floor():
    from descend_safety import (
        check_descend_tracking,
        descend_floor_z,
        ee_tracking_errors,
    )

    errs = ee_tracking_errors([0.5, 0.1, 0.25], [0.51, 0.11, 0.24])
    assert errs["track_err_xy"] == pytest.approx(math.hypot(0.01, 0.01), abs=1e-9)
    assert errs["track_err_z"] == pytest.approx(0.01, abs=1e-9)

    # Healthy step: small lag OK
    info = check_descend_tracking(
        ee_cmd=[0.50, 0.10, 0.24],
        ee_meas=[0.501, 0.1005, 0.239],
        anchor_xy=[0.50, 0.10],
        max_track_xy=0.015,
        max_track_z=0.025,
        max_track_xyz=0.030,
        max_anchor_xy=0.018,
    )
    assert "ee_cmd" in info and "ee_measured" in info
    assert info["track_err_xy"] < 0.015

    # Large XY track error → abort (table-slam guard)
    with pytest.raises(RuntimeError, match="descend tracking abort"):
        check_descend_tracking(
            ee_cmd=[0.50, 0.10, 0.20],
            ee_meas=[0.53, 0.10, 0.20],
            anchor_xy=[0.50, 0.10],
            max_track_xy=0.015,
            max_track_z=0.025,
            max_track_xyz=0.030,
            max_anchor_xy=0.018,
        )

    # Cumulative anchor drift even if cmd followed soft impedance rebase
    with pytest.raises(RuntimeError, match="anchor"):
        check_descend_tracking(
            ee_cmd=[0.52, 0.10, 0.18],
            ee_meas=[0.521, 0.100, 0.179],
            anchor_xy=[0.50, 0.10],
            max_track_xy=0.015,
            max_track_z=0.025,
            max_track_xyz=0.030,
            max_anchor_xy=0.018,
        )

    # Z bounce / overshoot like reliable_pick_20260723_090438
    with pytest.raises(RuntimeError, match="\\|z\\|_track"):
        check_descend_tracking(
            ee_cmd=[0.61, 0.07, 0.21],
            ee_meas=[0.62, 0.05, 0.25],
            anchor_xy=[0.61, 0.07],
            max_track_xy=0.015,
            max_track_z=0.025,
            max_track_xyz=0.030,
            max_anchor_xy=0.018,
        )

    zf = descend_floor_z(
        approach_z=0.30,
        grasp_z=0.18,
        max_drop=0.12,
        workspace_z_min=0.082,
        grasp_slack=0.005,
    )
    # max(0.082, 0.30-0.12=0.18, 0.18-0.005=0.175) = 0.18
    assert zf == pytest.approx(0.18, abs=1e-9)

    zf2 = descend_floor_z(
        approach_z=0.22,
        grasp_z=0.10,
        max_drop=0.12,
        workspace_z_min=0.082,
        grasp_slack=0.005,
    )
    # max(0.082, 0.10, 0.095) = 0.10
    assert zf2 == pytest.approx(0.10, abs=1e-9)


def test_r_offset_maps_calib_pose_to_train_mid():
    from state_builder import DEFAULT_R_OFFSET, franka_quat_to_libero_aa

    quat = np.array(
        [0.08202644286001429, 0.9957090046148952, -0.03889198204786296, -0.01796202418509244]
    )
    aa = franka_quat_to_libero_aa(quat, r_offset=DEFAULT_R_OFFSET)
    np.testing.assert_allclose(aa, [2.3079637, -0.02086651, -0.23397529], atol=1e-5)


def test_flip180_concat_shape():
    from camera_layout import franka_dual_to_wam_image

    c1 = np.zeros((480, 640, 3), np.uint8)
    c2 = np.zeros((480, 640, 3), np.uint8)
    c1[10, 10] = [255, 0, 0]
    rgb, pil = franka_dual_to_wam_image(c1, c2, flip180=True)
    assert pil.size == (448, 224)
    assert rgb.shape == (224, 448, 3)

def test_clamp_limits_xyz():
    x = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    y = clamp_eef_delta(x)
    assert abs(y[0]) <= 0.02 + 1e-6
    assert abs(y[1]) <= 0.02 + 1e-6


def test_clamp_linf_preserves_direction():
    from limits import StepLimits

    # Same shape as starved pick: large anisotropic delta
    x = np.array([-0.095, 0.075, -0.158, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    lim = StepLimits(max_abs_xyz=0.03, max_abs_rot=0.04)
    per = clamp_eef_delta(x, lim, mode="per_axis")
    linf = clamp_eef_delta(x, lim, mode="linf")
    # per-axis collapses to equal-magnitude axes (wrong ratio)
    assert abs(abs(per[0]) - abs(per[2])) < 1e-6
    # linf keeps xyz direction
    ratio_in = x[0] / x[2]
    ratio_out = linf[0] / linf[2]
    np.testing.assert_allclose(ratio_out, ratio_in, rtol=1e-5)
    assert abs(linf[0]) <= 0.03 + 1e-6
    assert abs(linf[1]) <= 0.03 + 1e-6
    assert abs(linf[2]) <= 0.03 + 1e-6
    assert abs(linf[2]) == pytest.approx(0.03, abs=1e-6)


def test_workspace_reject():
    with pytest.raises(ValueError):
        reject_if_outside_workspace(np.array([-1.0, 0.0, 0.0]))


def test_arming_default_disarmed():
    g = ArmingGate()
    assert g.armed is False
    with pytest.raises(PermissionError):
        g.require_armed()


def test_arming_one_shot():
    g = ArmingGate()
    g.issue_token("t1")
    g.arm("t1")
    assert g.armed is True
    g.require_armed()
    # token consumed; cannot re-arm without new token
    with pytest.raises(PermissionError):
        g.arm("t1")


def test_fake_controller_records():
    c = FakeController()
    c.send({"a": 1})
    assert c.commands == [{"a": 1}]


def test_denorm_roundtrip_fixture(tmp_path: Path):
    stats = {
        "action": {
            "min": [-0.1] * 7,
            "max": [0.1] * 7,
        }
    }
    raw = np.zeros(7, dtype=np.float32)
    out = denormalize_actions(raw, stats)
    assert out.shape == (7,)
    np.testing.assert_allclose(out, 0.0, atol=1e-5)


def test_replay_disarmed_does_not_send(tmp_path: Path):
    stats = {"action": {"min": [-1.0] * 7, "max": [1.0] * 7}}
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    jsonl = tmp_path / "a.jsonl"
    jsonl.write_text(
        json.dumps({"i": 0, "action": [[[0.0] * 7]]}) + "\n",
        encoding="utf-8",
    )
    audit = replay_actions_jsonl(jsonl, stats_path, arm=False)
    assert audit[0]["sent"] is False
    assert audit[0]["armed"] is False
