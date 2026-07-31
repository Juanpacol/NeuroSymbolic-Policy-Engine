"""Collation glue between HatefulMemesDataset and a torch DataLoader."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def collate_hateful_memes(
    batch: list[dict[str, Any]],
) -> tuple[Tensor, list[str], Tensor]:
    """Collates HatefulMemesDataset items into a training-ready batch.

    Args:
        batch: a list of items as returned by
            ``HatefulMemesDataset.__getitem__``, with ``image`` already
            run through a CLIP ``preprocess`` transform (i.e. a tensor
            of shape ``(3, H, W)``, not a raw PIL image).

    Returns:
        A tuple ``(images, texts, labels)``: ``images`` stacked to shape
        ``(batch, 3, H, W)``, ``texts`` as a list of strings, and
        ``labels`` as a float tensor of shape ``(batch,)``.
    """
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    labels = torch.tensor([float(item["label"]) for item in batch])
    return images, texts, labels
