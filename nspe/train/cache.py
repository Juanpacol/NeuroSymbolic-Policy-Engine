"""Precomputed CLIP embedding cache for the training path.

CLIP is frozen in both the reasoner and the baseline, so re-encoding
every image on every epoch recomputes an identical tensor N times and
re-pays the Hateful Memes mirror's per-image download on the first pass.
Encoding the split once into a flat tensor turns each subsequent epoch
into a pure head/reasoner pass over cached features -- the difference
between an hour per epoch and seconds, which is what makes full-dataset
runs feasible on a free GPU session.

Cache files record which backbone produced them and
:class:`EmbeddingDataset` refuses a mismatch, since embeddings from two
different backbones are not interchangeable and the failure is otherwise
silent -- a wrong-backbone cache trains happily and only shows up as
inexplicably poor metrics.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from nspe.train.dataset import collate_hateful_memes


def cache_path(
    cache_dir: str | Path, split: str, model_name: str, pretrained: str
) -> Path:
    """Builds the cache filename for one (split, backbone) pair.

    Keyed on the backbone's name rather than its embedding width:
    ``ViT-B-32`` and ``ViT-B-16`` both have ``output_dim == 512``, so a
    width-keyed name silently serves one backbone's embeddings to
    another.

    Args:
        cache_dir: directory holding cache files.
        split: dataset split name.
        model_name: an ``open_clip`` model architecture name.
        pretrained: an ``open_clip`` pretrained tag.

    Returns:
        The path this (split, backbone) pair should be cached at.
    """
    slug = f"{model_name}_{pretrained}".replace("/", "-")
    return Path(cache_dir) / f"hateful_memes_{split}_{slug}.pt"


def precompute_embeddings(
    encoder: nn.Module,
    dataset: Dataset[dict[str, object]],
    path: str | Path,
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 4,
    model_name: str = "",
    pretrained: str = "",
) -> dict[str, Tensor | str]:
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
        model_name: ``open_clip`` architecture that produced these
            embeddings, recorded so readers can reject a mismatch.
        pretrained: ``open_clip`` pretrained tag, recorded likewise.

    Returns:
        A dict with ``embeddings`` of shape ``(n, 2 * embed_dim)``,
        ``labels`` of shape ``(n,)``, and the recorded backbone tags.
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

    cache: dict[str, Tensor | str] = {
        "embeddings": torch.cat(chunks),
        "labels": torch.cat(label_chunks),
        "model_name": model_name,
        "pretrained": pretrained,
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
        expect_model: if given, raise unless the cache was written by
            this ``open_clip`` architecture.
        expect_pretrained: if given, raise unless the cache was written
            with this pretrained tag.
    """

    def __init__(
        self,
        path: str | Path,
        limit: int | None = None,
        expect_model: str | None = None,
        expect_pretrained: str | None = None,
    ) -> None:
        cache = torch.load(path, weights_only=True)
        for key, expected in (
            ("model_name", expect_model),
            ("pretrained", expect_pretrained),
        ):
            found = cache.get(key, "")
            if expected is not None and found and found != expected:
                raise ValueError(
                    f"{path} was built with {key}={found!r}, expected "
                    f"{expected!r}. Delete it or pass a matching backbone."
                )
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
