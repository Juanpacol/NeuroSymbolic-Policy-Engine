"""Precomputed CLIP embedding cache for the training path.

CLIP is frozen in both the reasoner and the baseline, so re-encoding
every image on every epoch recomputes an identical tensor N times and
re-pays the Hateful Memes mirror's per-image download on the first pass.
Encoding the split once into a flat tensor turns each subsequent epoch
into a pure head/reasoner pass over cached features -- the difference
between an hour per epoch and seconds, which is what makes full-dataset
runs feasible on a free GPU session.

The cache is keyed by nothing: it is the caller's job to keep a separate
file per (split, CLIP architecture) pair, since embeddings from two
different backbones are not interchangeable.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from nspe.train.dataset import collate_hateful_memes


def precompute_embeddings(
    encoder: nn.Module,
    dataset: Dataset[dict[str, object]],
    path: str | Path,
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 4,
) -> dict[str, Tensor]:
    """Encodes a whole dataset once and writes the result to disk.

    Args:
        encoder: a module exposing ``encode(images, texts)`` -- either a
            :class:`~nspe.extractor.NeuroSymbolicLayer` or a
            :class:`~nspe.baselines.neural_classifier.NeuralBaselineClassifier`.
            Both produce the same frozen-CLIP features, so one cache
            file serves either model.
        dataset: a :class:`~nspe.data.hateful_memes.HatefulMemesDataset`
            constructed with the encoder's ``preprocess`` transform.
        path: destination file for the cache.
        batch_size: encoding batch size.
        device: ``"cpu"``, ``"mps"``, or ``"cuda"``.
        num_workers: dataloader workers. Image decoding and the mirror's
            lazy per-file download dominate the first pass, so more
            workers help well past the point where the GPU saturates.

    Returns:
        A dict with ``embeddings`` of shape ``(n, 2 * embed_dim)`` and
        ``labels`` of shape ``(n,)``.
    """
    encoder = encoder.to(device)
    encoder.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_hateful_memes,
        num_workers=num_workers,
    )

    chunks: list[Tensor] = []
    label_chunks: list[Tensor] = []
    with torch.no_grad():
        for images, texts, labels in loader:
            fused = encoder.encode(images.to(device), texts)
            chunks.append(fused.float().cpu())
            label_chunks.append(labels)

    cache = {
        "embeddings": torch.cat(chunks),
        "labels": torch.cat(label_chunks),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return cache


class EmbeddingDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset over embeddings written by :func:`precompute_embeddings`.

    Args:
        path: a cache file previously written by
            :func:`precompute_embeddings`.
        limit: if given, use only the first ``limit`` examples.
    """

    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        cache = torch.load(path, weights_only=True)
        self.embeddings: Tensor = cache["embeddings"]
        self.labels: Tensor = cache["labels"]
        if limit is not None:
            self.embeddings = self.embeddings[:limit]
            self.labels = self.labels[:limit]

    def __len__(self) -> int:
        """Number of cached examples."""
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Returns the ``idx``-th ``(embedding, label)`` pair."""
        return self.embeddings[idx], self.labels[idx]


def collate_embeddings(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, None, Tensor]:
    """Collates cached embeddings into a training-ready batch.

    Emits the same ``(inputs, aux, labels)`` triple shape as
    :func:`~nspe.train.dataset.collate_hateful_memes` so
    :func:`~nspe.train.loop.train_model` needs no branch for the cached
    path -- ``aux`` is simply ``None``, since a precomputed embedding
    already carries the text.

    Args:
        batch: a list of ``(embedding, label)`` pairs.

    Returns:
        A tuple ``(embeddings, None, labels)``.
    """
    embeddings = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return embeddings, None, labels
