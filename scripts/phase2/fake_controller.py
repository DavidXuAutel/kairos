"""Records commanded targets; never talks to hardware."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeController:
    commands: list[dict] = field(default_factory=list)

    def send(self, payload: dict) -> None:
        self.commands.append(dict(payload))
