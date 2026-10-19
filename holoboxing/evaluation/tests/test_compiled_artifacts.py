"""Compiled observations must match both the source and frozen controller."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from holoboxing.evaluation.runtime import compiled_artifacts as compiled
from holoboxing.evaluation.runtime.model_contract import sha256


class CompiledArtifactsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "compiled"
        self.directory.mkdir()
        self.frozen = {"controller.onnx": {"bytes": 1, "sha256": "fixture"}}
        for target, value in (
            ("DIRECTORY", self.directory),
            ("APP_ROOT", self.root),
        ):
            context = patch.object(compiled, target, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch(
            "holoboxing.evaluation.models.frozen_manifest",
            return_value={"files": self.frozen},
        )
        context.start()
        self.addCleanup(context.stop)
        compiled.manifest.cache_clear()
        compiled.schema.cache_clear()
        self.addCleanup(compiled.manifest.cache_clear)
        self.addCleanup(compiled.schema.cache_clear)

    def bundle(self):
        def make(root, name):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            # Hash-verification fixtures; these are not runnable ONNX models.
            path.write_text("fingerprinted test artifact")
            return dict(bytes=path.stat().st_size, sha256=sha256(path))

        sources = (
            "holoboxing/evaluation/runtime/fast_runtime.py",
            "holoboxing/evaluation/runtime/representation.py",
            "holoboxing/evaluation/runtime/reference.py",
            "holoboxing/evaluation/runtime/observation_export.py",
            "holomotion/src/motion_tracking/link_observation.py",
            "holomotion/src/motion_tracking/link_actor_packing.py",
        )
        record = dict(
            schema_version=1,
            frozen_files=self.frozen,
            files={
                name: make(self.directory, name)
                for name in compiled.REQUIRED_FILES
            },
            sources={name: make(self.root, name) for name in sources},
        )
        self.write_manifest(record)
        return record

    def write_manifest(self, record):
        (self.directory / "manifest.json").write_text(json.dumps(record))

    def test_verified_snapshot_is_accepted(self):
        record = self.bundle()
        self.assertEqual(compiled.manifest(), record)

    def test_partial_bundle_does_not_select_source_backend(self):
        self.assertTrue(compiled.available())
        with self.assertRaises(FileNotFoundError):
            compiled.manifest()

    def test_modified_observation_graph_is_rejected(self):
        self.bundle()
        (self.directory / "upper_features.onnx").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            compiled.manifest()

    def test_changed_canonical_source_is_rejected(self):
        self.bundle()
        (self.root / "holoboxing/evaluation/runtime/reference.py").write_text(
            "new schema"
        )
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            compiled.manifest()

    def test_changed_controller_identity_is_rejected(self):
        record = self.bundle()
        record["frozen_files"] = {"another-controller": {}}
        self.write_manifest(record)
        with self.assertRaisesRegex(ValueError, "another controller"):
            compiled.manifest()

    def test_source_paths_cannot_escape_snapshot(self):
        record = self.bundle()
        record["sources"]["../outside.py"] = {"bytes": 0, "sha256": ""}
        self.write_manifest(record)
        with self.assertRaisesRegex(ValueError, "stay in their root"):
            compiled.manifest()
