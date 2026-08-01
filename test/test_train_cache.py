"""Tests for nspe.train.cache: embedding precompute, reload, collation.

Uses a stub encoder instead of CLIP -- what matters here is that the
cache round-trips embeddings and labels in dataset order, and that the
cached path feeds train_model the same batch shape as the raw path.
"""

import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.data import DataLoader, Dataset

from nspe.train.cache import (
    EmbeddingDataset,
    collate_embeddings,
    precompute_embeddings,
)

_EMBED_DIM = 4


class _StubEncoder(nn.Module):
    """Encodes an image batch to a deterministic function of its content."""

    def encode(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        del texts
        return images.flatten(1)[:, :_EMBED_DIM]


class _StubDataset(Dataset):
    """Yields HatefulMemesDataset-shaped items with tensor images."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "id": idx,
            "image": torch.full((3, 2, 2), float(idx)),
            "text": f"caption {idx}",
            "label": idx % 2,
        }


class TestPrecomputeEmbeddings(TestCase):
    def test_cache_round_trips_in_dataset_order(self):
        dataset = _StubDataset(7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "cache.pt"
            cache = precompute_embeddings(
                _StubEncoder(), dataset, path, batch_size=3, num_workers=0
            )

            self.assertTrue(path.exists())
            self.assertEqual(cache["embeddings"].shape, (7, _EMBED_DIM))
            self.assertEqual(cache["labels"].shape, (7,))

            reloaded = EmbeddingDataset(path)
            self.assertEqual(len(reloaded), 7)
            for idx in range(7):
                embedding, label = reloaded[idx]
                self.assertEqual(embedding, torch.full((_EMBED_DIM,), float(idx)))
                self.assertEqual(label.item(), float(idx % 2))

    def test_limit_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            precompute_embeddings(
                _StubEncoder(), _StubDataset(10), path, batch_size=4, num_workers=0
            )
            self.assertEqual(len(EmbeddingDataset(path, limit=3)), 3)

    def test_collate_emits_train_model_batch_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            precompute_embeddings(
                _StubEncoder(), _StubDataset(6), path, batch_size=6, num_workers=0
            )
            loader = DataLoader(
                EmbeddingDataset(path), batch_size=2, collate_fn=collate_embeddings
            )
            inputs, aux, labels = next(iter(loader))

            self.assertEqual(inputs.shape, (2, _EMBED_DIM))
            self.assertIsNone(aux)
            self.assertEqual(labels.shape, (2,))


if __name__ == "__main__":
    run_tests()
