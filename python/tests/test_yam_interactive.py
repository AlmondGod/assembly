"""Check file-session recovery and final-state verification without Isaac Sim."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class FakeWorld:
    """Minimal world exposing whether an interactive session stopped."""

    def __init__(self):
        """Create a world that has not been stopped."""
        self.stopped = False

    async def stop_async(self):
        """Record that the session requested a stop."""
        self.stopped = True


class FakeArm:
    """Expose a static pose for an unused test arm."""

    @staticmethod
    def pose(path: str) -> np.ndarray:
        """Return an identity pose for the test path.

        Args:
            path: Ignored simulator path.

        Returns:
            Identity world transform.
        """
        return np.eye(4)


def test_session_recovers_syntax_errors_and_rechecks_final_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Invalid code never runs; finish rechecks rather than using a stale success."""
    world_module = ModuleType("isaacsim.core.api.world")
    setattr(world_module, "World", FakeWorld)
    yam_module = ModuleType("bricksim.cap.yam")
    setattr(yam_module, "YamArm", FakeArm)
    core_module = ModuleType("bricksim.core")
    setattr(core_module, "are_parts_connected", lambda *_: False)
    setattr(
        core_module,
        "compute_connection_transform",
        lambda **_: (np.array([1, 0, 0, 0]), np.zeros(3)),
    )
    for name, module in (
        ("isaacsim.core.api.world", world_module),
        ("bricksim.cap.yam", yam_module),
        ("bricksim.core", core_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "bricksim/cap/interactive.py"
    spec = importlib.util.spec_from_file_location("interactive_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    world = FakeWorld()
    session = module.InteractiveSession(
        tmp_path, world, {}, {0: "base"}, {}, SimpleNamespace(connections={})
    )
    calls = []

    def snapshot(index: int, error: str | None = None):
        calls.append((index, error))
        # The prior observation said success; the final observation does not.
        session.valid_connections = {i: index < 2 for i in range(1, 5)}
        return {"connections": {str(i): {"valid": index < 2} for i in range(1, 5)}}

    monkeypatch.setattr(session, "snapshot", snapshot)
    (tmp_path / "command-000.json").write_text(
        json.dumps({"kind": "motion", "program": "await robot.move("})
    )
    (tmp_path / "command-001.json").write_text(json.dumps({"kind": "finish"}))
    asyncio.run(session.run())
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "incomplete"
    assert session.events[0]["error"]
    assert calls[-1][0] == 2
    assert world.stopped
