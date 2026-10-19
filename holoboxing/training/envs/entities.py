"""Named selections used by the boxing rewards, independent of the simulator."""

from dataclasses import dataclass, field


@dataclass
class SceneEntityCfg:
    name: str
    body_ids: object = field(default_factory=lambda: slice(None))
    joint_ids: object = field(default_factory=lambda: slice(None))
    body_names: object = None
    joint_names: object = None
