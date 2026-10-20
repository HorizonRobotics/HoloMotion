"""Fast local feedback cannot be confused with official qualification."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from holoboxing.evaluation.cli import main


class ValidationCLITest(unittest.TestCase):
    def test_local_default_and_full_modes_select_the_intended_scope(self):
        for options, full in (
            ([], False),
            (["--benchmark"], True),
            (["--qualification", "official"], True),
        ):
            with (
                self.subTest(options=options),
                tempfile.TemporaryDirectory() as directory,
            ):
                with patch(
                    "holoboxing.evaluation.validation.validate_model",
                    return_value={"status": "PASS"},
                ) as validate:
                    result = main(
                        [
                            "validate", "model.onnx", "--output",
                            str(Path(directory) / "report.json"), *options,
                        ]
                    )
                self.assertEqual(result, 0)
                self.assertEqual(validate.call_args.kwargs["full_chain"], full)
