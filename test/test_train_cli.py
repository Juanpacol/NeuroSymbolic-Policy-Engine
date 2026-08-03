"""Tests for the parts of nspe.train.cli that don't need a real CLIP model.

Full end-to-end training is exercised on real data only in
docs/colab_h1_h3.md's runbook; this covers the loader-selection logic
that decides which splits get built at all.
"""

import argparse
import tempfile
from pathlib import Path

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from nspe.train import cli as train_cli


def _write_cache(path: Path, n: int = 4) -> None:
    torch.save(
        {
            "embeddings": torch.randn(n, 8),
            "labels": torch.randint(0, 2, (n,)).float(),
            "model_name": "ViT-L-14",
            "pretrained": "openai",
        },
        path,
    )


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        cache_dir="",
        clip_model="ViT-L-14",
        clip_pretrained="openai",
        batch_size=4,
        limit_train=None,
        limit_val=None,
    )
    return argparse.Namespace(**{**defaults, **overrides})


class TestCachedLoadersSplitSelection(TestCase):
    def test_requesting_only_validation_never_touches_train(self):
        """The --epochs 0 path this backs: never download the train split.

        precompute_embeddings is what does the (potentially expensive,
        per-image-download) encoding; if it's called for "train" here,
        that regression has come back.
        """
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(cache_dir=tmp)
            _write_cache(
                train_cli.cache_path(tmp, "validation", "ViT-L-14", "openai")
            )

            calls = []
            original = train_cli.precompute_embeddings

            def spy(encoder, dataset, path, **kwargs):
                calls.append(path)
                return original(encoder, dataset, path, **kwargs)

            train_cli.precompute_embeddings = spy
            try:
                (loader,) = train_cli._cached_loaders(
                    torch.nn.Linear(1, 1),
                    preprocess=None,
                    args=args,
                    splits=(("validation", None),),
                )
            finally:
                train_cli.precompute_embeddings = original

            self.assertEqual(calls, [])
            self.assertEqual(len(loader.dataset), 4)

    def test_default_splits_still_build_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(cache_dir=tmp)
            _write_cache(train_cli.cache_path(tmp, "train", "ViT-L-14", "openai"))
            _write_cache(
                train_cli.cache_path(tmp, "validation", "ViT-L-14", "openai")
            )

            loaders = train_cli._cached_loaders(
                torch.nn.Linear(1, 1), preprocess=None, args=args
            )

            self.assertEqual(len(loaders), 2)


if __name__ == "__main__":
    run_tests()
