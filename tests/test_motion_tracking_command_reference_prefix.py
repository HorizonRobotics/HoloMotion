import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holomotion.src.utils.reference_prefix import resolve_reference_tensor_key


class MotionTrackingCommandReferencePrefixTests(unittest.TestCase):
    def test_ft_ref_prefix_uses_filtered_tensor_when_present(self):
        resolved = resolve_reference_tensor_key(
            batch_tensors={"ft_ref_root_pos": SimpleNamespace()},
            base_key="root_pos",
            prefix="ft_ref_",
        )

        self.assertEqual(resolved, "ft_ref_root_pos")

    def test_ft_ref_prefix_requires_filtered_tensor(self):
        with self.assertRaises(KeyError):
            resolve_reference_tensor_key(
                batch_tensors={"root_pos": SimpleNamespace()},
                base_key="root_pos",
                prefix="ft_ref_",
            )

    def test_ref_prefix_falls_back_to_unprefixed_tensor(self):
        resolved = resolve_reference_tensor_key(
            batch_tensors={"root_pos": SimpleNamespace()},
            base_key="root_pos",
            prefix="ref_",
        )

        self.assertEqual(resolved, "root_pos")

    def test_ref_prefix_prefers_prefixed_tensor_when_present(self):
        resolved = resolve_reference_tensor_key(
            batch_tensors={
                "root_pos": SimpleNamespace(),
                "ref_root_pos": SimpleNamespace(),
            },
            base_key="root_pos",
            prefix="ref_",
        )

        self.assertEqual(resolved, "ref_root_pos")
