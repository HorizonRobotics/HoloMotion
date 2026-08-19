# Modular Unitree G1 assets

- `g1_urdf_composer.py`: select body, head, and hand and compose a
  29-DOF URDF/MJCF pair.
- `components/`: 5 body, 2 head, and 6 non-empty hand descriptions, including
  the BrainCo Revo2 dedicated wrist-hand assembly. The `none` hand choice
  requires no component file.
- `mesh_index.json`: resolve shared and modular geometry while storing
  duplicate mesh content only once.
- `manifest.json`: module selections, aliases, source commit, and generated
  model metadata.

The robot asset payload is distributed through the
[`HorizonRobotics/HoloMotion_assets`](https://huggingface.co/datasets/HorizonRobotics/HoloMotion_assets)
Git submodule. Initialize the submodule and its Git LFS objects before
composition. The composer automatically reads the component package from
`thirdparties/HoloMotion_assets/assets/robots/unitree/G1/modular`. Packaged
runtimes can set `HOLOMOTION_G1_MODULAR_ASSET_ROOT` to an equivalent staged
layout.

The asset package contains reusable component descriptions and the exact
STL/mesh dependency closure. Complete donor URDFs and the 66 generated robot
files are not duplicated: URDF/MJCF outputs are generated deterministically
from the component package.

The repository tests validate deterministic composition, the 29-DOF contract,
mesh closure, and MuJoCo loading for all generated variants.
