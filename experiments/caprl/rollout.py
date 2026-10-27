"""Reproducible scenes and isolated, physically verified policy evaluation."""

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from bricksim.cap.basket import Basket
from bricksim.cap.policy import Assembly

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "demos/assemblies/four_brick_tower.json"
MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TRAIN_SEEDS = tuple(range(32))
EVAL_SEEDS = tuple(range(1000, 1016))


@dataclass(frozen=True)
class Scenario:
    """One tower task with logical brick IDs shuffled among safe basket slots."""

    seed: int

    def poses(self) -> dict[int, np.ndarray]:
        """Return initial column-vector world transforms in metres.

        Returns:
            Part ID to 4 by 4 transform; these are spawn poses before settling.
        """
        slots = Basket().scatter(self.seed)
        order = np.random.default_rng(self.seed + 100000).permutation([1, 2, 3, 4])
        return {part: slots[int(slot)] for part, slot in enumerate(order, 1)}

    def prompt(self) -> str:
        """Describe the task and observations without supplying arm choices.

        Returns:
            Plain text with the complete bounded action contract and source poses.
        """
        poses = {
            part: np.round(pose[:3, 3], 5).tolist()
            for part, pose in self.poses().items()
        }
        edges = [
            {"stud": c.stud, "hole": c.hole}
            for c in Assembly(TARGET).connections.values()
        ]
        return (
            "Build a four-brick tower using two YAM arms. Return only Python code, "
            "without markdown or explanation. Only literal statements of the form "
            'await robot.assemble(stud=0, hole=1, arm="left") are permitted. '
            'Choose arm="left" or arm="right" for each action. '
            "Place every target brick exactly once, after its supporting brick. "
            "The left arm is based at world Y=-0.26 m and the right at Y=+0.26 m. "
            "Choose an arm that can approach each source on its own side. "
            "The controller handles grasping, placement offsets and verification. "
            "Baseplate 0 is already placed. Source positions are initial simulator "
            "observations [x,y,z] in metres, before settling: "
            f"{json.dumps(poses)}. Desired connections: {json.dumps(edges)}."
        )


@dataclass(frozen=True)
class Outcome:
    """Separate code validity, verified physical progress, and completion."""

    valid: bool
    completed: int
    success: bool
    status: str

    @property
    def reward(self) -> float:
        """Return physical progress plus completion bonus, or invalid-code penalty."""
        return self.completed / 4 + float(self.success) if self.valid else -0.25

    @classmethod
    def from_result(cls, result: dict, policy: str) -> "Outcome":
        """Validate simulator evidence against the exact requested action sequence.

        Args:
            result: Fresh result or progress JSON from the controlled simulator.
            policy: The original generated program, already validated before launch.

        Returns:
            A reward-bearing outcome; malformed evidence raises ValueError.
        """
        actions = Assembly(TARGET).parse_policy(policy)
        events = result.get("actions", [])
        if len(events) > len(actions):
            raise ValueError("More action events than requested")
        for event, (connection, arm) in zip(events, actions, strict=False):
            if (
                event.get("stud") != connection.stud
                or event.get("hole") != connection.hole
                or event.get("arm") != arm
                or event.get("connected") is not True
                or not 0.065 <= event.get("pick", {}).get("lift_m", -1) < 0.5
            ):
                raise ValueError("Action evidence does not match verified execution")
        status = result["status"]
        if status not in ("success", "failed", "timeout"):
            raise ValueError(f"Nonterminal simulator status: {status}")
        success = status == "success"
        if success and (
            len(events) != 4
            or result.get("connections") != {str(i): True for i in range(1, 5)}
        ):
            raise ValueError("Success requires all four native connections")
        return cls(True, len(events), success, status)


class SimulatorReward:
    """Launch one fresh simulator per valid program, serially, with bounded runtime."""

    def __init__(self, output: Path, timeout: float = 900):
        """Configure immutable per-rollout outputs and a wall-clock limit.

        Args:
            output: Parent directory for unique rollout directories.
            timeout: Seconds allowed for each simulator, including startup.
        """
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        (ROOT / ".cache").mkdir(exist_ok=True)
        self.records: list[dict] = []
        self.__name__ = "physical_completion"

    def run(self, policy: str, seed: int) -> Outcome:
        """Evaluate a generated program without executing arbitrary Python.

        Args:
            policy: Literal bounded action program; invalid programs never launch.
            seed: Deterministic source placement and ID-permutation seed.

        Returns:
            Physical completion and progress. Infrastructure failures raise.
        """
        directory = self.output / f"seed-{seed}-{uuid4().hex}"
        directory.mkdir()
        (directory / "policy.py").write_text(policy)
        (directory / "request.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "prompt": Scenario(seed).prompt(),
                    "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
                    "target_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
                },
                indent=2,
            )
        )
        try:
            Assembly(TARGET).parse_policy(policy)
        except (ValueError, SyntaxError) as error:
            outcome = Outcome(False, 0, False, "invalid_policy")
            (directory / "invalid.txt").write_text(str(error))
        else:
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("YAM_")
            }
            environment.update(
                YAM_OUTPUT=str(directory),
                YAM_POLICY=str(directory / "policy.py"),
                YAM_SEED=str(seed),
                YAM_TARGET=str(TARGET),
                YAM_BASKET="1",
                YAM_SUPPLY_COUNT="4",
                YAM_SCENE_ONLY="0",
                PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "python")]),
            )
            command = [
                str(ROOT / ".venv/bin/bricksim"),
                "--/app/window/enabled=false",
                "--/isaac/startup/ros_bridge_extension=",
                str(ROOT / "experiments/caprl/simulate.py"),
            ]
            # The lock protects concurrent training/evaluation workers, not other demos.
            with (ROOT / ".cache/caprl.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with (directory / "simulator.log").open("w") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    started = time.monotonic()
                    result = {}
                    try:
                        while True:
                            result_path = directory / "result.json"
                            if result_path.exists():
                                try:
                                    result = json.loads(result_path.read_text())
                                except json.JSONDecodeError:
                                    result = {}  # A small file may be mid-write.
                                if result.get("status") in ("success", "failed"):
                                    break
                            if process.poll() is not None:
                                raise RuntimeError(f"Simulator exited: {directory}")
                            if time.monotonic() - started >= self.timeout:
                                progress = directory / "progress.json"
                                if not progress.exists():
                                    raise RuntimeError(f"Startup timeout: {directory}")
                                result = json.loads(progress.read_text())
                                result.update(status="timeout")
                                result_path.write_text(json.dumps(result, indent=2))
                                break
                            time.sleep(1)
                    finally:
                        # Isaac can leave child processes even after its launcher exits.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
            if not (directory / "inventory.json").exists():
                raise RuntimeError(f"Scene initialization failed: {directory}")
            if result.get("status") == "failed":
                failure = directory / "failure.json"
                if not failure.exists() or not json.loads(failure.read_text()).get(
                    "expected_controller_failure", False
                ):
                    raise RuntimeError(f"Unexpected simulator failure: {directory}")
            outcome = Outcome.from_result(result, policy)
        record = dict(
            directory=str(directory),
            seed=seed,
            valid=outcome.valid,
            completed=outcome.completed,
            success=outcome.success,
            status=outcome.status,
            reward=outcome.reward,
        )
        (directory / "outcome.json").write_text(json.dumps(record, indent=2))
        with (self.output / "rollouts.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.records.append(record)
        print(json.dumps(record), flush=True)
        return outcome

    def __call__(
        self, completions: list[str], scene_seed: list[int], **kwargs: object
    ) -> list[float | None]:
        """Provide TRL's custom reward callback for plain-text completions.

        Args:
            completions: Sampled programs, grouped by prompt by TRL.
            scene_seed: Dataset seed repeated for each generated completion.
            **kwargs: Additional TRL metadata, unused by physical verification.

        Returns:
            One physically measured scalar reward per completion.
        """
        return [
            self.run(code, seed).reward
            for code, seed in zip(completions, scene_seed, strict=True)
        ]
