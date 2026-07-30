"""Neural predicate extractor: frozen CLIP + trainable per-predicate heads.

The only trainable parameters are the linear heads. CLIP stays frozen
so the extractor is cheap to train (no backbone fine-tuning) and its
image/text embeddings remain comparable across policies. Gradients
still flow from a verdict, through the reasoner, through the heads --
just not into CLIP itself.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NeuroSymbolicLayer(nn.Module):
    """Maps (image, text) pairs to base predicate truth degrees.

    Args:
        predicate_names: base predicate names, in the order the
            reasoner expects them (i.e.
            ``policy.predicate_names("base")``).
        model_name: an ``open_clip`` model architecture name.
        pretrained: an ``open_clip`` pretrained tag for ``model_name``.
    """

    def __init__(
        self,
        predicate_names: tuple[str, ...],
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
        self.predicate_names = predicate_names

        embed_dim = self.clip.visual.output_dim
        self.heads = nn.Linear(embed_dim * 2, len(predicate_names))

    @classmethod
    def from_policy(
        cls,
        policy: object,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
    ) -> NeuroSymbolicLayer:
        """Builds an extractor sized for a policy's base predicates.

        Args:
            policy: a :class:`~nspe.policy.schema.Policy`.
            model_name: an ``open_clip`` model architecture name.
            pretrained: an ``open_clip`` pretrained tag for
                ``model_name``.

        Returns:
            A :class:`NeuroSymbolicLayer` with one output per base
            predicate in ``policy``.
        """
        base_names = policy.predicate_names("base")  # type: ignore[attr-defined]
        return cls(base_names, model_name=model_name, pretrained=pretrained)

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
        with torch.no_grad():
            image_features = self.clip.encode_image(images)
            tokens = self.tokenizer(texts).to(images.device)
            text_features = self.clip.encode_text(tokens)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return torch.cat([image_features, text_features], dim=-1)

    def forward(self, images: Tensor, texts: list[str]) -> Tensor:
        """Produces base predicate truth degrees for a batch.

        Args:
            images: preprocessed image batch, shape
                ``(batch, 3, H, W)``.
            texts: one caption/text string per image, length ``batch``.

        Returns:
            Tensor of shape ``(batch, len(predicate_names))``, values in
            ``(0, 1)``, suitable as ``mu0`` for
            :class:`~nspe.reasoner.PolicyKGReasoner`.
        """
        fused = self.encode(images, texts)
        return torch.sigmoid(self.heads(fused))
