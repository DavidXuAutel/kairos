"""One-shot arming gate. Default disarmed. Does not create ROS pubs."""
from __future__ import annotations


class ArmingGate:
    def __init__(self) -> None:
        self._armed = False
        self._token: str | None = None

    @property
    def armed(self) -> bool:
        return self._armed

    def issue_token(self, token: str) -> None:
        if not token:
            raise ValueError("empty arm token")
        self._token = token
        self._armed = False

    def arm(self, token: str) -> None:
        if self._token is None or token != self._token:
            raise PermissionError("invalid or missing arm token")
        self._armed = True
        self._token = None  # one-shot

    def disarm(self) -> None:
        self._armed = False
        self._token = None

    def require_armed(self) -> None:
        if not self._armed:
            raise PermissionError("controller disarmed")
