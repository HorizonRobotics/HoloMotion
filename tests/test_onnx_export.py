import sys
from types import SimpleNamespace

from holomotion.src.utils.onnx_export import attach_onnx_metadata_holomotion


class _FakeEntry:
    def __init__(self):
        self.key = ""
        self.value = ""


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        return _FakeTensor(self._values[index])

    def cpu(self):
        return self

    def tolist(self):
        return self._values


def test_attach_onnx_metadata_uses_default_joint_gains(monkeypatch):
    model = SimpleNamespace(metadata_props=[])
    fake_onnx = SimpleNamespace(
        load=lambda path: model,
        save=lambda loaded_model, path: None,
        StringStringEntryProto=_FakeEntry,
    )
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

    robot_data = SimpleNamespace(
        joint_names=["joint_a", "joint_b"],
        joint_stiffness=_FakeTensor([[0.0, 0.0]]),
        joint_damping=_FakeTensor([[0.0, 0.0]]),
        default_joint_stiffness=_FakeTensor([[10.0, 20.0]]),
        default_joint_damping=_FakeTensor([[1.0, 2.0]]),
        default_joint_pos=_FakeTensor([[0.1, -0.2]]),
    )
    action_term = SimpleNamespace(
        _joint_names=["joint_b", "joint_a"],
        _scale=_FakeTensor([[0.25, 0.5]]),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=robot_data)},
        action_manager=SimpleNamespace(
            get_term=lambda name: action_term,
        ),
    )

    attach_onnx_metadata_holomotion(env, "dummy.onnx")

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    assert metadata["joint_names"] == "joint_b,joint_a"
    assert metadata["joint_stiffness"] == "20.000,10.000"
    assert metadata["joint_damping"] == "2.000,1.000"
    assert metadata["default_joint_pos"] == "-0.200,0.100"
    assert metadata["action_scale"] == "0.250,0.500"
