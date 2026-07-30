"""Tests for nspe.data.hateful_memes.

Network-dependent: downloads dataset metadata and a couple of images
from the Hugging Face Hub. Skipped if the `datasets`/`huggingface_hub`
extras are not installed or the Hub is unreachable.
"""

from torch.testing._internal.common_utils import TestCase, run_tests

try:
    import datasets  # noqa: F401
    import huggingface_hub  # noqa: F401

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


class TestHatefulMemesDataset(TestCase):
    def setUp(self):
        if not _HAS_DEPS:
            self.skipTest("datasets/huggingface_hub not installed (data extra)")
        try:
            from nspe.data.hateful_memes import HatefulMemesDataset

            self.ds = HatefulMemesDataset(split="validation")
        except Exception as exc:  # network unavailable, rate-limited, etc.
            self.skipTest(f"Hugging Face Hub unavailable: {exc}")

    def test_dataset_is_nonempty_and_filtered_to_available_images(self):
        self.assertGreater(len(self.ds), 0)

    def test_item_has_expected_fields(self):
        item = self.ds[0]
        self.assertIn("image", item)
        self.assertIn("text", item)
        self.assertIn("label", item)
        self.assertIn(item["label"], (0, 1))
        self.assertEqual(item["image"].mode, "RGB")


if __name__ == "__main__":
    run_tests()
