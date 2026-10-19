"""Exercise the host launcher without starting Docker or touching a GPU."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[2] / "run.sh"


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        binary = self.root / "bin"
        binary.mkdir()
        fake = binary / "docker"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json,os,sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1:3] == ['image','inspect']:\n"
            "    if '--format' in sys.argv: print(os.environ.get('DOCKER_CONTRACT', 'urdf-gloves-v1 epa-plane-fp64-v1'))\n"
            "    sys.exit(0)\n"
            "Path(os.environ['DOCKER_CALL']).write_text(json.dumps(sys.argv[1:]))\n"
        )
        fake.chmod(0o755)
        self.log = self.root / "docker.json"
        self.environment = dict(
            os.environ,
            PATH=str(binary) + os.pathsep + os.environ["PATH"],
            DOCKER_CALL=str(self.log),
            HOLOBOXING_IMAGE="provided:image",
        )

    def run_launcher(self, *args):
        return subprocess.run(
            ["bash", str(LAUNCHER), *map(str, args)],
            cwd=self.root,
            env=self.environment,
            text=True,
            capture_output=True,
        )

    def model(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model passed through without host execution")
        return path

    def test_image_without_both_fixes_is_rejected_before_starting(self):
        self.environment["DOCKER_CONTRACT"] = "<no value> <no value>"
        replay = self.model("replay.hdf5")
        result = self.run_launcher("--view", replay)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("predates", result.stderr)
        self.assertFalse(self.log.exists())

    def test_model_paths_are_literal_and_outputs_are_writable(self):
        model = self.model("models/red $(touch injected).onnx")
        output = self.root / "results with spaces" / "validation.json"
        result = self.run_launcher("--validate", model, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.log.read_text())
        self.assertIn(
            f"type=bind,src={model},dst=/inputs/red.onnx,readonly", args
        )
        self.assertIn(f"type=bind,src={output.parent},dst=/results", args)
        self.assertTrue(output.parent.is_dir())
        self.assertFalse((self.root / "injected").exists())
        self.assertIn("device=0", args)
        self.assertIn("none", args)
        self.assertNotIn("--assets", args)
        self.assertNotIn("--lower", args)

    def test_init_placeholder_explains_manual_setup_without_docker(self):
        result = self.run_launcher("--init")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Pull the published image", result.stderr)
        self.assertIn("No installation or download", result.stderr)
        self.assertFalse(self.log.exists())
        self.assertFalse((self.root / "results").exists())

    def test_match_keeps_same_named_files_from_different_directories(self):
        red, blue = (
            self.model("red/policy.onnx"),
            self.model("blue/policy.onnx"),
        )
        result = self.run_launcher("--match", red, blue)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.log.read_text())
        self.assertIn(
            f"type=bind,src={red},dst=/inputs/red.onnx,readonly", args
        )
        self.assertIn(
            f"type=bind,src={blue},dst=/inputs/blue.onnx,readonly", args
        )

    def test_view_needs_no_gpu_and_binds_only_localhost(self):
        replay = self.model("saved match/replay.hdf5")
        result = self.run_launcher("--view", replay, "--port", "8081")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.log.read_text())
        self.assertNotIn("--gpus", args)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", args)
        self.assertIn("127.0.0.1:8081:8080", args)
        self.assertIn(
            f"type=bind,src={replay},dst=/inputs/replay.hdf5,readonly", args
        )

    def test_existing_results_are_not_replaced_or_started(self):
        model = self.model("red.onnx")
        output = self.model("saved/validation.json")
        original = output.read_bytes()
        result = self.run_launcher("--validate", model, "--output", output)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Output already exists", result.stderr)
        self.assertEqual(output.read_bytes(), original)
        self.assertFalse(self.log.exists())

    def test_conflicting_or_invalid_options_fail_before_docker(self):
        for args in (
            ("--init", "--view", "one"),
            ("--validate",),
            ("--match", "one"),
            ("--validate", "one", "--view", "two"),
            ("--view", "one", "--gpu", "0"),
            ("--match", "one", "two", "--port", "8081"),
            ("--view", "one", "--port", "65536"),
            ("--validate", "one", "--gpu", "-1"),
        ):
            with self.subTest(args=args):
                result = self.run_launcher(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(self.log.exists())

    def test_print_command_has_no_output_directory_side_effect(self):
        model = self.model("red.onnx")
        output = self.root / "new directory" / "validation.json"
        result = self.run_launcher(
            "--validate", model, "--output", output, "--print-command"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(shlex.split(result.stdout)[0:2], ["docker", "run"])
        self.assertFalse(output.parent.exists())
        self.assertFalse(self.log.exists())
