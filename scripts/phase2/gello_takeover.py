"""Take over /gello/joint_states by pausing gello_publisher and streaming goals."""
from __future__ import annotations

import os
import signal
import threading
import time
from typing import Optional

import numpy as np

from hw_helpers import find_gello_publisher_pids

JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]


class GelloTakeover:
    """Pause gello_publisher and publish joint goals at fixed rate."""

    def __init__(self, node, rate_hz: float = 50.0) -> None:
        from sensor_msgs.msg import JointState

        self._node = node
        self._JointState = JointState
        self._pub = node.create_publisher(JointState, "/gello/joint_states", 10)
        self._rate_hz = float(rate_hz)
        self._lock = threading.Lock()
        self._q = np.zeros(7, dtype=np.float64)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._paused_pids: list[int] = []

    def set_goal(self, q: np.ndarray) -> None:
        with self._lock:
            self._q = np.asarray(q, dtype=np.float64).reshape(7).copy()

    def _loop(self) -> None:
        period = 1.0 / self._rate_hz
        while self._running:
            t0 = time.time()
            with self._lock:
                q = self._q.copy()
            msg = self._JointState()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.name = list(JOINT_NAMES)
            msg.position = [float(x) for x in q]
            self._pub.publish(msg)
            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

    def start(self, q0: np.ndarray) -> None:
        self.set_goal(q0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(0.2)
        self._paused_pids = []
        for pid in find_gello_publisher_pids():
            try:
                os.kill(pid, signal.SIGKILL)
                self._paused_pids.append(pid)
                print(f"[gello] SIGKILL pid={pid}", flush=True)
            except ProcessLookupError:
                pass

    def stop(self, resume_gello: bool = False) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if resume_gello:
            for pid in self._paused_pids:
                try:
                    os.kill(pid, signal.SIGCONT)
                    print(f"[gello] SIGCONT pid={pid}", flush=True)
                except ProcessLookupError:
                    pass
        else:
            print("[gello] left STOPPED (no SIGCONT)", flush=True)
        self._paused_pids = []
