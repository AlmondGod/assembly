"""Verify physical-reward evidence and observation-conditioned training tasks."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from experiments.caprl.rollout import (
    EVAL_SEEDS,
    ROOT,
    TRAIN_SEEDS,
    Outcome,
    Scenario,
    SimulatorReward,
)

POLICY = "\n".join(
    f"await robot.assemble(stud={i - 1}, hole={i}, "
    f'arm="{"left" if i % 2 else "right"}")'
    for i in range(1, 5)
)


@pytest.fixture
def evidence() -> dict:
    """Return existing physically verified evidence for reward-contract tests."""
    return json.loads((ROOT / "docs/yam/basket/result-seed7.json").read_text())


def test_physical_success_and_partial_progress(evidence: dict):
    """Only four native connections receive the completion bonus."""
    full = Outcome.from_result(evidence, POLICY)
    assert full.reward == 2 and full.success and full.completed == 4
    evidence.update(status="failed", actions=evidence["actions"][:2])
    partial = Outcome.from_result(evidence, POLICY)
    assert partial.reward == 0.5 and not partial.success
    evidence["status"] = "timeout"
    assert Outcome.from_result(evidence, POLICY).reward == 0.5


@pytest.mark.parametrize("defect", ["lift", "connection", "order", "arm", "status"])
def test_reject_inconsistent_evidence(evidence: dict, defect: str):
    """Missing native verification or mismatched actions cannot earn success."""
    result = copy.deepcopy(evidence)
    if defect == "lift":
        result["actions"][0]["pick"]["lift_m"] = 0.001
    elif defect == "connection":
        result["connections"]["4"] = False
    elif defect == "order":
        result["actions"].reverse()
    elif defect == "arm":
        result["actions"][0]["arm"] = "right"
    else:
        result["status"] = "running"
    with pytest.raises(ValueError):
        Outcome.from_result(result, POLICY)


def test_invalid_policy_never_launches_simulator(tmp_path: Path):
    """Arbitrary Python receives a distinct invalid-code result before launch."""
    reward = SimulatorReward(tmp_path)
    result = reward.run('import os\nos.system("echo forbidden")', 7)
    assert not result.valid and not result.success and result.reward == -0.25
    assert not list(tmp_path.glob("*/simulator.log"))
    assert len(reward.records) == 1


def test_split_and_reproducible_arm_variation():
    """Held-out seeds are disjoint and IDs do not encode the optimal arm."""
    assert not set(TRAIN_SEEDS) & set(EVAL_SEEDS)
    side_assignments = set()
    for seed in TRAIN_SEEDS:
        poses = Scenario(seed).poses()
        repeated = Scenario(seed).poses()
        assert set(poses) == {1, 2, 3, 4}
        for part in poses:
            np.testing.assert_array_equal(poses[part], repeated[part])
        side_assignments.add(tuple(pose[1, 3] > 0 for pose in poses.values()))
        prompt = Scenario(seed).prompt()
        assert "Alternate left" not in prompt
        assert "initial simulator" in prompt
    assert len(side_assignments) == 6


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "timeout",
        "startup_failure",
        "crash",
        "physical_failure",
        "runtime_failure",
    ],
)
def test_subprocess_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evidence: dict, mode: str
):
    """Use a fixture subprocess to verify timeout, cleanup and fail-fast behavior."""
    import os
    import sys

    from experiments.caprl import rollout

    root = tmp_path / "fixture-runtime"
    executable = root / ".venv/bin/bricksim"
    executable.parent.mkdir(parents=True)
    # This is explicitly a process-control fixture, not a physical simulator.
    script = f"""#!{sys.executable}
import json, os, time
from pathlib import Path
out = Path(os.environ['YAM_OUTPUT'])
(out / 'pid').write_text(str(os.getpid()))
mode = {mode!r}
assert 'YAM_BASKET_LAYOUT' not in os.environ
assert os.environ['YAM_SUPPLY_COUNT'] == '4'
result = {evidence!r}
if mode == 'crash':
    raise RuntimeError('fixture startup crash')
if mode == 'startup_failure':
    (out / 'result.json').write_text(json.dumps({{'status': 'failed'}}))
elif mode in ('physical_failure', 'runtime_failure'):
    (out / 'inventory.json').write_text('{{}}')
    result.update(status='failed', actions=result['actions'][:2])
    (out / 'failure.json').write_text(json.dumps({{
        'expected_controller_failure': mode == 'physical_failure'}}))
    (out / 'result.json').write_text(json.dumps(result))
elif mode == 'timeout':
    (out / 'inventory.json').write_text('{{}}')
    result.update(status='running', actions=result['actions'][:2])
    (out / 'progress.json').write_text(json.dumps(result))
else:
    (out / 'inventory.json').write_text('{{}}')
    (out / 'result.json').write_text(json.dumps(result))
time.sleep(60)
"""
    executable.write_text(script)
    executable.chmod(0o755)
    monkeypatch.setenv("YAM_BASKET_LAYOUT", "compact")
    monkeypatch.setattr(rollout, "ROOT", root)
    reward = SimulatorReward(tmp_path / "rollouts", timeout=1.5)
    if mode in ("startup_failure", "crash", "runtime_failure"):
        with pytest.raises(RuntimeError):
            reward.run(POLICY, 7)
        assert not reward.records
    else:
        outcome = reward.run(POLICY, 7)
        assert outcome.reward == (2 if mode == "success" else 0.5)
        assert outcome.status == ("failed" if mode == "physical_failure" else mode)
    pid_path = next((tmp_path / "rollouts").glob("*/pid"))
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)
