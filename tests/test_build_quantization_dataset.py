import importlib.util
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "not_for_commit"
        / "build_quantization_dataset.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_quantization_dataset", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_allocate_sample_counts_normalizes_weights_and_matches_total():
    module = _load_module()

    counts = module.allocate_sample_counts(
        {
            "AMASS": 0.2,
            "lafan1": 0.2,
            "MotionMillion-ft": 0.4,
            "pico_train": 0.05,
        },
        17,
    )

    assert counts == {
        "AMASS": 4,
        "lafan1": 4,
        "MotionMillion-ft": 8,
        "pico_train": 1,
    }
    assert sum(counts.values()) == 17


def test_build_quantization_dataset_creates_symlinks(tmp_path):
    module = _load_module()
    npz_root = tmp_path / "retargeted"
    npz_root.mkdir()

    for dataset_name, clip_count in {"AMASS": 3, "lafan1": 2}.items():
        dataset_dir = npz_root / dataset_name
        dataset_dir.mkdir()
        for clip_idx in range(clip_count):
            (dataset_dir / f"clip_{clip_idx}.npz").write_text(
                f"{dataset_name}-{clip_idx}", encoding="utf-8"
            )

    output_dir = module.build_quantization_dataset(
        npz_root=npz_root,
        dataset_ratios={"AMASS": 2.0, "lafan1": 1.0},
        num_clips=3,
        seed=0,
        current_date="20260324",
    )

    assert output_dir == npz_root / "20260324_quant_dataset"
    created_links = sorted(output_dir.iterdir())
    assert len(created_links) == 3
    assert all(link.is_symlink() for link in created_links)
    assert {link.name.split("__", 1)[0] for link in created_links} == {
        "AMASS",
        "lafan1",
    }
    for link in created_links:
        assert link.resolve().is_file()
        assert link.suffix == ".npz"


def test_build_quantization_dataset_caps_each_dataset_at_available_clips(
    tmp_path,
):
    module = _load_module()
    npz_root = tmp_path / "retargeted"
    npz_root.mkdir()

    for dataset_name, clip_count in {"AMASS": 1, "lafan1": 5}.items():
        dataset_dir = npz_root / dataset_name
        dataset_dir.mkdir()
        for clip_idx in range(clip_count):
            (dataset_dir / f"clip_{clip_idx}.npz").write_text(
                f"{dataset_name}-{clip_idx}", encoding="utf-8"
            )

    output_dir = module.build_quantization_dataset(
        npz_root=npz_root,
        dataset_ratios={"AMASS": 2.0, "lafan1": 1.0},
        num_clips=6,
        seed=0,
        current_date="20260324",
    )

    created_links = sorted(output_dir.iterdir())

    assert len(created_links) == 3
    assert sum(link.name.startswith("AMASS__") for link in created_links) == 1
    assert sum(link.name.startswith("lafan1__") for link in created_links) == 2
