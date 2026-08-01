"""End-to-end neural baseline: the H1/H3 comparison point for PolicyEngine.

Mirrors NeuroSymbolicLayer's frozen-CLIP feature path exactly (via the
shared `_clip_fused_embedding` helper) but replaces the per-predicate
heads with a single linear head straight to a verdict. Same features,
same capacity, no interpretable predicate layer -- so any consistency or
accuracy difference observed against the symbolic reasoner is
attributable to the reasoning step, not to a difference in the neural
backbone.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nspe.extractor import _clip_fused_embedding


class NeuralBaselineClassifier(nn.Module):
    """Maps (image, text) pairs directly to a single verdict truth degree.

    Args:
        model_name: an ``open_clip`` model architecture name.
        pretrained: an ``open_clip`` pretrained tag for ``model_name``.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
    ) -> None:
        super().__init__()
        import open_clip  # type: ignore[import-untyped]

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

        self.preprocess = preprocess
        self.tokenizer = tokenizer

        embed_dim = self.clip.visual.output_dim
        self.head = nn.Linear(embed_dim * 2, 1)

    def encode(self, images: Tensor, texts: list[str]) -> Tensor:
        """Encodes preprocessed images and raw text into a fused embedding.

        Args:
            images: preprocessed image batch, shape
                ``(batch, 3, H, W)``, as produced by ``self.preprocess``.
            texts: one caption/text string per image, length ``batch``.

        Returns:
            L2-normalized, concatenated image+text embeddings, shape
            ``(batch, 2 * embed_dim)``.
        """
        return _clip_fused_embedding(self.clip, self.tokenizer, images, texts)

    def forward_embedded(self, fused: Tensor) -> Tensor:
        """Produces a verdict from a precomputed fused embedding.

        Args:
            fused: fused CLIP embeddings, shape
                ``(batch, 2 * embed_dim)``, as returned by
                :meth:`encode`.

        Returns:
            Tensor of shape ``(batch,)``, values in ``(0, 1)``.
        """
        return torch.sigmoid(self.head(fused)).squeeze(-1)

    def forward(self, images: Tensor, texts: list[str]) -> Tensor:
        """Produces a single verdict truth degree per case.

        Args:
            images: preprocessed image batch, shape ``(batch, 3, H, W)``.
            texts: one caption/text string per image, length ``batch``.

        Returns:
            Tensor of shape ``(batch,)``, values in ``(0, 1)``, directly
            comparable to a :class:`~nspe.reasoner.PolicyKGReasoner`
            verdict.
        """
        return self.forward_embedded(self.encode(images, texts))
