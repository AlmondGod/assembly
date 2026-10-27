# Eleven-brick heart with two YAM arms

The heart uses six red 2×4 bricks and five red 2×2 bricks, plus a 20×20 support baseplate. Its six-layer arrangement follows the [published eleven-brick heart](https://www.steamsational.com/lego-valentines/). The eight-stud width follows world Y; the two-stud depth follows world X. Exactly two YAM arms pick from organized tabletop positions and alternate physical placements, bottom-up.

Generate a fresh policy with the separate Qwen environment described in [the YAM setup](README.md):

```sh
PYTHONPATH=python .venv-qwen/bin/python demos/generate_yam_policy.py \
  --target demos/assemblies/eleven_brick_heart.json \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output outputs/yam-heart
PYTHONPATH=.:python uv run --no-sync bricksim \
  --/app/window/enabled=false --/isaac/startup/ros_bridge_extension= \
  demos/demo_yam_heart.py
```

The dedicated entrypoint selects the heart target, output directory and reachable source layout. `YAM_POLICY` can select an existing validated heart program. `YAM_SCENE_ONLY=1` previews the initial inventory. Do not set `YAM_BASKET` for this scene. The grounded model must place IDs 1–11 in order and alternate left/right arms, beginning with left; the entrypoint rejects a different sequence before simulation because the organized source positions are assigned to those arms.

The baseplate starts at X=0.26 m so both arms can reach the high sixth layer with 100 mm approach clearance. Source parts start disconnected. The scene never updates their poses during assembly: fingers must lift each brick, physical contact must create each required connection, and the controller must verify relative placement within 2 mm and 5 degrees. Each new brick has one declared anchor support; touching neighboring supports may create additional native connections.

This is a simulation demonstration with simulator-pose observations, organized stock and sequential arm movements. It does not demonstrate arbitrary-pile picking, camera localization, hardware-calibrated forces or a general collision planner. A real cell would require those capabilities and a fixture supporting the narrow heart base.

`inventory.json`, `result.json`, the raw Qwen response and generation provenance record what ran. `yam-scene.png` shows the initial two-arm scene; `yam-tower.png` is a broad-side close-up of the heart despite its historical filename. Each `step-N.png` follows a verified placement.

```sh
uv run --no-sync pytest --noconftest -q python/tests/test_yam_heart.py \
  python/tests/test_yam_policy.py python/tests/test_yam_kinematics.py \
  python/tests/test_yam_basket.py
uv run --no-sync ruff check
uv run --no-sync ty check
```

The geometry tests reconstruct all six layers from the declared stud offsets, check six 2×4 plus five 2×2 bricks without overlap, and require actual supporting stud intersections. Reachability tests cover each assigned arm's source, lifted source, placement and raised placement poses.
