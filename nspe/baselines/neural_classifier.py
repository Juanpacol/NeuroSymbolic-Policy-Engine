"""End-to-end neural baseline: the H1/H3 comparison point for PolicyEngine.

Mirrors NeuroSymbolicLayer exactly up to the aggregation step: the same
frozen-CLIP feature path (via the shared `_clip_fused_embedding`
helper), the same `PredicateTrunk`, and the same number of latent
predicate units. The one difference is what turns those units into a
verdict -- a fixed, auditable policy circuit for the reasoner, a learned
linear aggregator here. The baseline is therefore *the reasoner with the
policy replaced by a learned aggregator over the same latent predicate
vector*, which is what makes an observed H1/H3 difference attributable
to the reasoning step.

An earlier version compared a single `Linear(1024, 1)` (1025 parameters)
against `Linear(1024, 6)` plus a logic circuit (6150 parameters) while
claiming matched capacity. It was not matched, and any difference it
measured was confounded.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nspe.calibration import VerdictCalibrator
from nspe.extractor import _clip_fused_embedding
from nspe.trunk import PredicateHead


class NeuralBaselineClassifier(nn.Module):
    """Maps (image, text) pairs directly to a single verdict truth degree.

    Args:
        model_name: an ``open_clip`` model architecture name.
        pretrained: an ``open_clip`` pretrained tag for ``model_name``.
        num_predicates: width of the latent predicate layer. Set this to
            the policy's base predicate count so the trunk and head
            shapes match the reasoner arm exactly.
        hidden_dim: width of the shared trunk.
        dropout: trunk dropout probability.
        calibrator: optional monotone map applied to the verdict, so
            both arms of the comparison receive identical treatment.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        num_predicates: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        calibrator: VerdictCalibrator | None = None,
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

        fused_dim = self.clip.visual.output_dim * 2
        self.head = PredicateHead(
            fused_dim, num_predicates, hidden_dim=hidden_dim, dropout=dropout
        )
        self.aggregator = nn.Linear(num_predicates, 1)
        self.calibrator = calibrator

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
        verdict = torch.sigmoid(self.aggregator(self.head(fused))).squeeze(-1)
        return verdict if self.calibrator is None else self.calibrator(verdict)

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
