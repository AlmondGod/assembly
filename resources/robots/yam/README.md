# YAM v1 model

URDF and meshes from [I2RT Robotics](https://github.com/i2rt-robotics/i2rt/tree/120c3c81400171174604e503943f8d1ebc891058/i2rt/robot_models/arm/yam/v1), commit `120c3c81400171174604e503943f8d1ebc891058`. The included MIT license applies to these files. Original files are unmodified.

Six revolute arm joints use radians; two prismatic finger joints use metres. The flange has +Z forward and +X across the fingers. The importer generates convex mesh collisions from visuals. Simulation drive gains and force limits are configured by the demo; source effort limits are placeholders, and this is not a calibrated hardware controller.
