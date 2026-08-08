"""Neural predicate extractor: frozen CLIP + trainable predicate heads.

The only trainable parameters are the trunk, the heads and their
scaling. CLIP stays frozen so the extractor is cheap to train (no
backbone fine-tuning) and its image/text embeddings remain comparable
across policies. Gradients still flow from a verdict, through the
reasoner, through the heads -- just not into CLIP itself.

Two design choices exist to stop the predicate layer collapsing. All
heads read the same embedding and receive gradient from a single scalar
verdict, with nothing in the loss pushing them apart, so they are free
to become correlated copies of one label predictor: a run of this model
produced only 5 distinct thresholded predicate signatures across 831
cases, out of 64 possible. The residual zero-shot path gives each
predicate a distinct, semantically grounded gradient path derived from
its own natural-language definition in the policy, and the per-predicate
logit scale gives the optimizer direct control over how sharply each
predicate splits the batch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nspe.trunk import PredicateHead

# open_clip ships no type stubs, so a constructed model/tokenizer is
# seen as Any; these describe just the surface this module relies on,
# so a caller who casts an open_clip object to one gets real checking
# on every attribute actually used below.
Preprocess = Callable[[Any], Any]
ClipTokenizer = Callable[[list[str]], Tensor]


class _ClipVisualTower(Protocol):
    """The subset of open_clip's visual tower this module reads."""

    output_dim: int


class ClipModel(Protocol):
    """The open_clip model surface `_clip_fused_embedding` depends on."""

    visual: _ClipVisualTower

    def encode_image(self, images: Tensor) -> Tensor:
        """Encodes a preprocessed image batch into CLIP's visual space."""
        ...

    def encode_text(self, tokens: Tensor) -> Tensor:
        """Encodes tokenized text into CLIP's shared embedding space."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yields the model's parameters, for the freeze-on-load loop."""
        ...

    def eval(self) -> ClipModel:
        """Switches to inference mode; returns self, as `nn.Module` does."""
        ...


class Encoder(Protocol):
    """The frozen-CLIP-encoder interface `precompute_embeddings` needs.

    Both :class:`NeuroSymbolicLayer` and
    :class:`~nspe.baselines.neural_classifier.NeuralBaselineClassifier`
    satisfy this structurally, which is what lets one cache-building
    function serve either arm.
    """

    preprocess: Preprocess

    def encode(self, images: Tensor, texts: list[str]) -> Tensor:
        """Encodes a preprocessed image/text batch into a fused embedding."""
        ...

    def to(self, device: str) -> Encoder:
        """Moves the encoder to `device`; returns self, as `nn.Module` does."""
        ...

    def eval(self) -> Encoder:
        """Switches to inference mode; returns self, as `nn.Module` does."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Yields the encoder's parameters."""
        ...


def _clip_fused_embedding(
    clip: ClipModel, tokenizer: ClipTokenizer, images: Tensor, texts: list[str]
) -> Tensor:
    """Encodes preprocessed images and raw text into a fused CLIP embedding.

    Shared by :class:`NeuroSymbolicLayer` and
    :class:`~nspe.baselines.neural_classifier.NeuralBaselineClassifier` so
    both consume the exact same frozen-CLIP feature path -- the H1/H3
    comparison is only fair if the two models differ solely in what sits
    on top of these features, not in how the features themselves are
    computed.

    Args:
        clip: a frozen ``open_clip`` model exposing ``encode_image`` and
            ``encode_text``.
        tokenizer: the ``open_clip`` tokenizer matching ``clip``.
        images: preprocessed image batch, shape ``(batch, 3, H, W)``.
        texts: one caption/text string per image, length ``batch``.

    Returns:
        L2-normalized, concatenated image+text embeddings, shape
        ``(batch, 2 * embed_dim)``.
    """
    with torch.no_grad():
        image_features = clip.encode_image(images)
        tokens = tokenizer(texts).to(images.device)
        text_features = clip.encode_text(tokens)
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    return torch.cat([image_features, text_features], dim=-1)


class NeuroSymbolicLayer(nn.Module):
    """Maps (image, text) pairs to base predicate truth degrees.

    Args:
        predicate_names: base predicate names, in the order the
            reasoner expects them (i.e.
            ``policy.predicate_names("base")``).
        model_name: an ``open_clip`` model architecture name.
        pretrained: an ``open_clip`` pretrained tag for ``model_name``.
        hidden_dim: width of the shared :class:`~nspe.trunk.PredicateTrunk`.
            ``0`` reproduces the linear-probe configuration.
        dropout: trunk dropout probability.
        mu_eps: half-width of the margin kept away from 0 and 1. The
            reasoner floors truth degrees at ``eps`` and ``log1mexp``
            clamps its input, so a head that saturates contributes
            exactly zero gradient through every negated occurrence of
            its predicate -- which, for predicates appearing only under
            ``unless:``, is their only occurrence. Emitting a strictly
            interior range keeps both clamps off the live path.
    """

    def __init__(
        self,
        predicate_names: tuple[str, ...],
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        mu_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        import open_clip  # type: ignore[import-untyped]

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        self.clip: ClipModel = cast(ClipModel, clip_model)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

        self.preprocess: Preprocess = preprocess
        self.tokenizer: ClipTokenizer = cast(ClipTokenizer, tokenizer)
        self.predicate_names = predicate_names
        self.mu_eps = mu_eps

        fused_dim = self.clip.visual.output_dim * 2
        self.head = PredicateHead(
            fused_dim,
            len(predicate_names),
            hidden_dim=hidden_dim,
            dropout=dropout,
            mu_eps=mu_eps,
        )

    @classmethod
    def from_policy(
        cls,
        policy: object,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "openai",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        init_from_descriptions: bool = True,
    ) -> NeuroSymbolicLayer:
        """Builds an extractor sized for a policy's base predicates.

        Args:
            policy: a :class:`~nspe.policy.schema.Policy`.
            model_name: an ``open_clip`` model architecture name.
            pretrained: an ``open_clip`` pretrained tag for
                ``model_name``.
            hidden_dim: width of the shared trunk.
            dropout: trunk dropout probability.
            init_from_descriptions: if ``True``, seed the zero-shot
                residual path from each base predicate's
                natural-language ``description`` in the policy.

        Returns:
            A :class:`NeuroSymbolicLayer` with one output per base
            predicate in ``policy``.
        """
        base_names = policy.predicate_names("base")  # type: ignore[attr-defined]
        layer = cls(
            base_names,
            model_name=model_name,
            pretrained=pretrained,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        if init_from_descriptions:
            descriptions = policy.predicate_descriptions("base")  # type: ignore[attr-defined]
            layer.init_heads_from_descriptions(descriptions)
        return layer

    @torch.no_grad()
    def init_heads_from_descriptions(
        self,
        descriptions: dict[str, str],
        prompt: str = "a meme where {}",
    ) -> None:
        """Seeds the zero-shot residual from policy predicate descriptions.

        Encodes each predicate's natural-language definition with the
        frozen CLIP text encoder and stores the L2-normalized result,
        duplicated across the image and text halves of the fused
        embedding. Each predicate therefore has a distinct, semantically
        grounded contribution to its own logit from step 0, which breaks
        the symmetry that otherwise lets every head converge on the same
        direction. This is also the only per-predicate supervision
        available at all: Hateful Memes carries no predicate labels, so
        the grounding has to come from the policy text itself.

        Predicates missing from ``descriptions``, or with an empty one,
        keep a zero residual and rely solely on the learned head.

        Args:
            descriptions: predicate name to description text.
            prompt: template wrapped around each description before
                encoding, matching CLIP's caption-like training
                distribution.
        """
        wanted = [
            (i, descriptions.get(name, "").strip())
            for i, name in enumerate(self.predicate_names)
        ]
        present = [(i, text) for i, text in wanted if text]
        if not present:
            return

        weight = self.head.zero_shot_weight
        tokens = self.tokenizer([prompt.format(t) for _, t in present]).to(
            weight.device
        )
        features = F.normalize(self.clip.encode_text(tokens), dim=-1)
        # The fused embedding is [image_half, text_half]; a description
        # is a text query against both halves, so it is duplicated.
        rows = torch.cat([features, features], dim=-1)
        for row, (index, _) in enumerate(present):
            weight[index] = rows[row].to(weight.dtype)

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
        """Produces predicate truth degrees from a precomputed embedding.

        Skips the frozen CLIP backbone entirely, so training can run off
        embeddings cached once by
        :func:`~nspe.train.cache.precompute_embeddings` instead of
        re-encoding every image on every epoch.

        Args:
            fused: fused CLIP embeddings, shape
                ``(batch, 2 * embed_dim)``, as returned by
                :meth:`encode`.

        Returns:
            Tensor of shape ``(batch, len(predicate_names))``, values in
            ``[mu_eps, 1 - mu_eps]``.
        """
        # nn.Module.__call__ is typed to return Any.
        return cast(Tensor, self.head(fused))

    def forward(self, images: Tensor, texts: list[str]) -> Tensor:
        """Produces base predicate truth degrees for a batch.

        Args:
            images: preprocessed image batch, shape
                ``(batch, 3, H, W)``.
            texts: one caption/text string per image, length ``batch``.

        Returns:
            Tensor of shape ``(batch, len(predicate_names))``, values in
            ``[mu_eps, 1 - mu_eps]``, suitable as ``mu0`` for
            :class:`~nspe.reasoner.PolicyKGReasoner`.
        """
        return self.forward_embedded(self.encode(images, texts))
