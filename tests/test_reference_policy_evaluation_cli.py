import json
import subprocess

import pytest

from holomotion.scripts.evaluation import eval_reference_policy as cli


def test_input_relocation_preserves_contents_and_rejects_drift(tmp_path):
    root = tmp_path / "relocated"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"hdf5_shards": [{"file": "motion.h5"}]}))
    robot = root / "robot.urdf"
    robot.write_text(
        '<robot><link><visual><geometry><mesh filename="m.stl"/></geometry></visual></link></robot>'
    )
    (root / "motion.h5").write_bytes(b"frozen motion")
    (root / "m.stl").write_bytes(b"frozen mesh")
    hashes = {
        "/original/" + p.name: cli.fingerprint(p) for p in root.iterdir()
    }
    spec = {"input_sha256": hashes}
    assert len(cli.verify_inputs(spec, manifest, robot)) == 4
    (root / "motion.h5").write_bytes(b"different motion")
    with pytest.raises(ValueError, match="inputs differ"):
        cli.verify_inputs(spec, manifest, robot)


def test_task_source_drift_is_rejected(tmp_path):
    paths = (
        "holomotion/src/env/reference_tracking/task.py",
        "holomotion/src/algo/supermimic/networks.py",
        "holomotion/src/training/h5_dataloader.py",
        "holomotion/src/training/reference_fk.py",
    )
    for name in paths:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    spec = {"source_sha256": {p: cli.fingerprint(tmp_path / p) for p in paths}}
    assert len(cli.verify_task_source(spec, tmp_path)) == 4
    (tmp_path / paths[0]).write_text("changed task")
    with pytest.raises(ValueError, match="task source changed"):
        cli.verify_task_source(spec, tmp_path)
    with pytest.raises(ValueError, match="lacks"):
        cli.verify_task_source({"source_sha256": {}}, tmp_path)


def test_parent_uses_separate_interpreter_and_checks_completion(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "evaluation"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        (output / "report.json").write_text('{"completed": true}')

    monkeypatch.setattr(cli.subprocess, "run", run)
    cli.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--mode",
            "canonical_gpu_v1",
            "--timeout-seconds",
            "123",
        ]
    )
    command, kwargs = calls[0]
    assert command[:3] == [
        cli.sys.executable,
        "-m",
        "holomotion.scripts.evaluation.eval_reference_policy",
    ]
    assert command[-1] == "--worker"
    assert kwargs["timeout"] == 123 and kwargs["check"] is True
    assert json.loads((output / "run_status.json").read_text())["completed"]


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired("worker", 1),
        subprocess.CalledProcessError(1, "worker"),
    ],
)
def test_worker_failure_never_becomes_success(tmp_path, monkeypatch, failure):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "evaluation"

    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(type(failure)):
        cli.main(["--checkpoint", str(checkpoint), "--output", str(output)])
    assert (
        json.loads((output / "run_status.json").read_text())["completed"]
        is False
    )


def test_existing_results_are_preserved(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(SystemExit):
        cli.main(["--checkpoint", str(checkpoint), "--output", str(tmp_path)])
    assert checkpoint.read_bytes() == b"checkpoint"
