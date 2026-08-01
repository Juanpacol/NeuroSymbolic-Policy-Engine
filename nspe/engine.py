"""End-to-end wiring: neural extractor -> symbolic reasoner."""

from __future__ import annotations

from torch import Tensor, nn

from nspe.extractor import NeuroSymbolicLayer
from nspe.reasoner import PolicyKGReasoner, ReasonerOutput


class PolicyEngine(nn.Module):
    """Runs an extractor and a reasoner as one differentiable pipeline.

    Gradients flow from a verdict, through the reasoner's rules, through
    the extractor's predicate heads -- the reasoner's rule structure is
    exactly what supervises which visual/textual cues each head should
    learn to respond to, without ever needing per-predicate labels.

    Args:
        extractor: produces base predicate truth degrees from raw
            (image, text) pairs.
        reasoner: consumes those truth degrees and produces verdicts.
    """

    def __init__(
        self, extractor: NeuroSymbolicLayer, reasoner: PolicyKGReasoner
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.reasoner = reasoner

    def forward_embedded(self, fused: Tensor) -> ReasonerOutput:
        """Runs the pipeline from a precomputed fused embedding.

        Args:
            fused: fused CLIP embeddings, shape
                ``(batch, 2 * embed_dim)``.

        Returns:
            The reasoner's :class:`~nspe.reasoner.ReasonerOutput`.
        """
        out: ReasonerOutput = self.reasoner(self.extractor.forward_embedded(fused))
        return out

    def forward(self, images: Tensor, texts: list[str]) -> ReasonerOutput:
        """Runs the full pipeline on one batch.

        Args:
            images: preprocessed image batch, shape
                ``(batch, 3, H, W)``.
            texts: one caption/text string per image, length ``batch``.

        Returns:
            The reasoner's :class:`~nspe.reasoner.ReasonerOutput`.
        """
        mu0 = self.extractor(images, texts)
        out: ReasonerOutput = self.reasoner(mu0)
        return out
