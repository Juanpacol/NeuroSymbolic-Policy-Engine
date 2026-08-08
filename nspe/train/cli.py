"""Training CLI: fits either the reasoner path or the neural baseline.

Usage:
    python -m nspe.train.cli --model reasoner --device cuda \
        --cache-dir ./emb_cache --out checkpoints/reasoner.pt
    python -m nspe.train.cli --model baseline --device cuda \
        --cache-dir ./emb_cache --out checkpoints/baseline.pt

Both are trained identically: BCE on the "hateful" verdict against
Hateful Memes' real binary label. For --model reasoner, gradients flow
through the differentiable PolicyKGReasoner into the extractor's
predicate heads; for --model baseline, only the head is trained. CLIP
stays frozen in both cases.

Because CLIP is frozen, --cache-dir avoids re-encoding the split on
every epoch: pass a directory and the first run encodes each split once
to disk, while that run and every later one train off those cached
embeddings. On a free GPU session this is the difference between an
epoch dominated by image downloads and an epoch that takes seconds.
Cache files are keyed by backbone and refuse to load under a different
one.

Both arms carry a VerdictCalibrator whose bias is fitted to the training
base rate before the first epoch, and checkpoints are selected on
validation AUROC rather than BCE -- see nspe/calibration.py and
nspe/train/loop.py for why selecting on BCE rewarded a constant model.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from nspe.baselines.neural_classifier import NeuralBaselineClassifier
from nspe.calibration import VerdictCalibrator
from nspe.engine import PolicyEngine
from nspe.extractor import Encoder, NeuroSymbolicLayer, Preprocess
from nspe.policy.loader import load_policy
from nspe.reasoner import PolicyKGReasoner
from nspe.train.cache import (
    EmbeddingDataset,
    cache_path,
    collate_embeddings,
    precompute_embeddings,
)
from nspe.train.dataset import collate_hateful_memes
from nspe.train.loop import train_model
from nspe.train.regularizers import (
    activation_entropy_loss,
    anchor_loss,
    decorrelation_loss,
    zero_shot_targets,
)
from nspe.train.seed import set_seed

_VERDICT_NAME = "hateful"
# What train_model's forward_fn parameter expects; the four concrete
# functions below are each narrower in their first argument, which
# Callable's parameter contravariance means is not directly assignable
# -- see where each is cast to this at its point of use in train_one.
_ForwardFn = Callable[[nn.Module, Tensor, Any], Tensor]


def _verdict_of(out: object) -> Tensor:
    """Prefers the calibrated verdict, falling back to the raw one."""
    calibrated = getattr(out, "calibrated", None)
    if calibrated is not None:
        return cast(Tensor, calibrated[_VERDICT_NAME])
    return cast(Tensor, out.verdicts[_VERDICT_NAME])  # type: ignore[attr-defined]


def _reasoner_forward(model: PolicyEngine, images: Tensor, texts: list[str]) -> Tensor:
    return _verdict_of(model(images, texts))


def _reasoner_forward_embedded(model: PolicyEngine, fused: Tensor, _: None) -> Tensor:
    return _verdict_of(model.forward_embedded(fused))


def _baseline_forward(
    model: NeuralBaselineClassifier, images: Tensor, texts: list[str]
) -> Tensor:
    # nn.Module.__call__ is typed to return Any.
    return cast(Tensor, model(images, texts))


def _baseline_forward_embedded(
    model: NeuralBaselineClassifier, fused: Tensor, _: None
) -> Tensor:
    return model.forward_embedded(fused)


def _build_model(args: argparse.Namespace) -> tuple[nn.Module, Preprocess]:
    """Builds the model under test and its CLIP preprocessing transform.

    Both arms are sized from the same policy so their trunk and latent
    predicate layer match exactly; only the aggregation differs.
    """
    policy = load_policy(args.policy)
    num_predicates = len(policy.predicate_names("base"))
    calibrator = VerdictCalibrator()

    if args.model == "reasoner":
        extractor = NeuroSymbolicLayer.from_policy(
            policy,
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            init_from_descriptions=not args.no_description_init,
        )
        reasoner = PolicyKGReasoner(
            policy,
            store_trace=False,
            learnable_confidence=args.learnable_confidence,
            aggregate=args.aggregate,
            pmean_p=args.pmean_p,
        )
        engine = PolicyEngine(extractor, reasoner, calibrator=calibrator)
        return engine, extractor.preprocess

    model = NeuralBaselineClassifier(
        model_name=args.clip_model,
        pretrained=args.clip_pretrained,
        num_predicates=num_predicates,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        calibrator=calibrator,
    )
    return model, model.preprocess


def _make_aux_loss(
    model: nn.Module, args: argparse.Namespace
) -> Callable[[nn.Module, Tensor, object], Tensor] | None:
    """Builds the anti-collapse auxiliary loss, or ``None`` if disabled.

    Only applies to the reasoner arm: the baseline's latent units are
    not predicates and carry no interpretability claim, so regularizing
    them toward the policy's semantics would be meaningless.
    """
    if not isinstance(model, PolicyEngine):
        return None
    weights = (args.lambda_anchor, args.lambda_decorr, args.lambda_entropy)
    if not any(w > 0 for w in weights):
        return None

    extractor = model.extractor

    def aux(engine: nn.Module, inputs: Tensor, side: Any) -> Tensor:
        # The cached path hands over fused embeddings directly; the raw
        # path hands over images that still need encoding.
        fused = inputs if args.cache_dir is not None else extractor.encode(inputs, side)
        mu0 = extractor.forward_embedded(fused)
        loss = fused.sum() * 0.0
        if args.lambda_anchor > 0:
            targets = zero_shot_targets(fused, extractor.head.zero_shot_weight)
            loss = loss + args.lambda_anchor * anchor_loss(mu0, targets)
        if args.lambda_decorr > 0:
            loss = loss + args.lambda_decorr * decorrelation_loss(mu0)
        if args.lambda_entropy > 0:
            loss = loss + args.lambda_entropy * activation_entropy_loss(mu0)
        return loss

    return aux


def _encoder_of(model: nn.Module) -> Encoder:
    # Both branches satisfy Encoder structurally (NeuroSymbolicLayer and
    # NeuralBaselineClassifier each define .preprocess/.encode), but
    # model's own type here is the broader nn.Module the two arms share.
    return cast(Encoder, model.extractor if isinstance(model, PolicyEngine) else model)


def _cached_loaders(
    model: nn.Module,
    preprocess: Preprocess,
    args: argparse.Namespace,
    splits: tuple[tuple[str, int | None], ...] | None = None,
) -> list[DataLoader[Any]]:
    """Builds loaders over cached embeddings, encoding splits if needed.

    Args:
        model: the model whose encoder produces the cached embeddings.
        preprocess: the encoder's image transform.
        args: this CLI's namespace.
        splits: which ``(split, limit)`` pairs to build, in order;
            defaults to train and validation. A caller that only needs
            validation (``--epochs 0``, see :func:`train_one`) passes
            just that, so the train split's images are never downloaded
            or encoded at all -- building it eagerly encodes the whole
            split via :func:`~nspe.train.cache.precompute_embeddings`,
            which for the Hateful Memes mirror means fetching several
            thousand images one at a time.

    Returns:
        One :class:`~torch.utils.data.DataLoader` per requested split,
        in the same order.
    """
    encoder = _encoder_of(model)
    loaders = []
    for split, limit in splits or (
        ("train", args.limit_train),
        ("validation", args.limit_val),
    ):
        path = cache_path(args.cache_dir, split, args.clip_model, args.clip_pretrained)
        if not path.exists():
            from nspe.data.hateful_memes import HatefulMemesDataset

            print(f"Encoding {split} split -> {path} (one-time)")
            precompute_embeddings(
                encoder,
                HatefulMemesDataset(split=split, transform=preprocess),
                path,
                batch_size=args.batch_size,
                device=args.device,
                model_name=args.clip_model,
                pretrained=args.clip_pretrained,
            )
        dataset = EmbeddingDataset(
            path,
            limit=limit,
            expect_model=args.clip_model,
            expect_pretrained=args.clip_pretrained,
        )
        if limit is not None:
            _require_both_classes(dataset.labels.tolist(), split, limit)
        print(f"{split}: {len(dataset)} cached examples")
        loaders.append(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=split == "train",
                collate_fn=collate_embeddings,
            )
        )
    return loaders


def _require_both_classes(labels: Sequence[float], split: str, limit: int) -> None:
    """Rejects a truncated split that ended up single-class.

    The Hateful Memes mirror orders its rows by label, so a head-of-split
    subset is one class only. On the validation split that silently
    breaks checkpoint *selection*, not just a reported number: AUROC is
    0.5 for every epoch, so ``--select-metric auroc`` keeps whichever
    epoch happened to come first.

    Args:
        labels: labels of the truncated split.
        split: split name, for the error message.
        limit: the ``--limit-*`` value that produced the truncation.

    Raises:
        ValueError: if every label is the same class.
    """
    positive = sum(1 for label in labels if label > 0.5)
    if positive in (0, len(labels)):
        raise ValueError(
            f"--limit-{split} {limit} leaves a single class "
            f"({positive}/{len(labels)} positive): this dataset is ordered "
            "by label, so a head-of-split subset is degenerate. Use the "
            "full split."
        )


def _raw_loaders(
    preprocess: Preprocess, args: argparse.Namespace
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Builds loaders that encode images on the fly, every epoch."""
    from nspe.data.hateful_memes import HatefulMemesDataset

    loaders = []
    for split, limit in (("train", args.limit_train), ("validation", args.limit_val)):
        full_dataset = HatefulMemesDataset(split=split, transform=preprocess)
        dataset: Dataset[Any] = full_dataset
        if limit is not None:
            kept = min(limit, len(full_dataset))
            _require_both_classes(full_dataset.labels()[:kept], split, limit)
            dataset = Subset(full_dataset, range(kept))
        loaders.append(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=split == "train",
                collate_fn=collate_hateful_memes,
            )
        )
    return loaders[0], loaders[1]


def _calibrator_of(model: nn.Module) -> VerdictCalibrator | None:
    return getattr(model, "calibrator", None)


@torch.no_grad()
def _warm_start(
    model: nn.Module,
    forward_fn: _ForwardFn,
    loader: Iterable[tuple[Tensor, Any, Tensor]],
    device: str,
) -> tuple[float, float]:
    """Fits the calibrator bias to the training base rate.

    Runs one untrained pass to see where the uncalibrated verdict
    distribution actually sits, then shifts the calibrator so the mean
    predicted probability equals the label base rate. The model
    therefore starts at the constant solution instead of having to
    descend toward it, which is what previously consumed training.

    Args:
        model: the model about to be trained.
        forward_fn: the batch-to-verdict callable used in training.
        loader: the training loader, or an empty iterable if there is no
            training to warm-start for (``--epochs 0``).
        device: device to run the pass on.

    Returns:
        A tuple ``(base_rate, pos_weight)`` for the training split.
    """
    model = model.to(device)
    model.eval()
    calibrator = _calibrator_of(model)
    raw, targets = [], []
    for inputs, aux, labels in loader:
        if calibrator is not None:
            calibrator.enabled = False
        raw.append(forward_fn(model, inputs.to(device), aux).cpu())
        targets.append(labels)
    if calibrator is not None:
        calibrator.enabled = True

    labels = torch.cat(targets)
    base_rate = labels.mean().item()
    if calibrator is not None and 0.0 < base_rate < 1.0:
        calibrator.fit_bias_to_base_rate(torch.cat(raw), base_rate)
    num_pos = labels.sum().item()
    pos_weight = (labels.numel() - num_pos) / max(num_pos, 1.0)
    return base_rate, pos_weight


def build_parser() -> argparse.ArgumentParser:
    """Builds the training CLI parser.

    Exposed so callers that drive training programmatically -- the
    ablation sweep in :mod:`nspe.ablate.cli` -- can render a
    configuration as an argument list and parse it here, instead of
    hand-building a ``Namespace``. Every default and ``choices=``
    constraint then stays in sync with a manual run by construction,
    rather than by somebody remembering to update two places.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["reasoner", "baseline"])
    parser.add_argument(
        "--policy",
        default="nspe/policies/hateful_memes.yaml",
        help="Path to a policy YAML file (ignored for --model baseline).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Shared trunk width. 0 reproduces the linear-probe ablation.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--no-description-init",
        action="store_true",
        help="Skip seeding the zero-shot residual from the policy's "
        "predicate descriptions (ablation).",
    )
    parser.add_argument(
        "--learnable-confidence",
        action="store_true",
        help="Learn rule confidences instead of using the policy's "
        "declared ones. Ablation only: the primary result should come "
        "from the confidences the published policy actually states.",
    )
    parser.add_argument(
        "--lambda-anchor",
        type=float,
        default=0.1,
        help="Weight on the CLIP-description anchor loss. Too large and "
        "the predicate layer degenerates into CLIP zero-shot.",
    )
    parser.add_argument("--lambda-decorr", type=float, default=0.05)
    parser.add_argument("--lambda-entropy", type=float, default=0.02)
    parser.add_argument(
        "--aggregate",
        default="tconorm",
        choices=["tconorm", "pmean"],
        help="How rules sharing a head combine. Ablation only: a "
        "t-conorm is what makes the output a fuzzy-logic verdict.",
    )
    parser.add_argument(
        "--pmean-p",
        type=float,
        default=2.0,
        help="Power for --aggregate pmean. 1 is the arithmetic mean, "
        "larger approaches max.",
    )
    parser.add_argument(
        "--select-metric",
        default="auroc",
        choices=["auroc", "f1", "accuracy", "bce"],
        help="Validation metric driving checkpointing and early stopping. "
        "Selecting on bce rewards a model that collapses to the base rate.",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--no-class-weight",
        action="store_true",
        help="Disable the positive-class up-weight in the BCE term.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory of precomputed CLIP embeddings. Splits missing "
        "from it are encoded once and reused by later runs.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint to load before training, e.g. --out from an interrupted run.",
    )
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Write the per-epoch metric history to this JSON file.",
    )
    return parser


def train_one(args: argparse.Namespace) -> dict[str, Any]:
    """Runs one training configuration end to end.

    Args:
        args: a namespace from :func:`build_parser`.

    Returns:
        The :func:`~nspe.train.loop.train_model` result dict with an
        ``args`` entry, matching what ``--metrics-out`` writes. Writing
        that file is left to the caller, so a sweep can own its own
        output without racing this one.
    """
    set_seed(args.seed)
    model, preprocess = _build_model(args)

    train_loader: Iterable[tuple[Tensor, Any, Tensor]]
    val_loader: DataLoader[Any]
    forward_fn: _ForwardFn

    if args.cache_dir is not None:
        # With epochs=0 nothing ever iterates the train split, so
        # requesting only validation here means it is never encoded --
        # for the Hateful Memes mirror, building it eagerly would
        # otherwise download several thousand images for nothing.
        splits = (
            (("validation", args.limit_val),)
            if args.epochs == 0
            else (("train", args.limit_train), ("validation", args.limit_val))
        )
        loaders = _cached_loaders(model, preprocess, args, splits=splits)
        if args.epochs == 0:
            train_loader, val_loader = [], loaders[0]
        else:
            train_loader, val_loader = loaders
        forward_fn = cast(
            _ForwardFn,
            _reasoner_forward_embedded
            if args.model == "reasoner"
            else _baseline_forward_embedded,
        )
    else:
        train_loader, val_loader = _raw_loaders(preprocess, args)
        forward_fn = cast(
            _ForwardFn,
            _reasoner_forward if args.model == "reasoner" else _baseline_forward,
        )

    if args.epochs == 0:
        # Nothing to warm-start a calibrator bias for.
        base_rate, pos_weight = 0.5, 1.0
    else:
        base_rate, pos_weight = _warm_start(
            model, forward_fn, train_loader, args.device
        )
        print(f"train base rate={base_rate:.4f} pos_weight={pos_weight:.3f}")

    result = train_model(
        model,
        forward_fn,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=None if args.no_class_weight else pos_weight,
        aux_loss_fn=_make_aux_loss(model, args),
        select_metric=args.select_metric,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.out,
        resume_from=args.resume,
    )
    return {"args": vars(args), **result}


def _print_result(args: argparse.Namespace, result: dict[str, Any]) -> None:
    """Prints the headline metrics for one training run."""
    best = result["history"][result["best_epoch"]]
    print(
        f"model={args.model} clip={args.clip_model} seed={args.seed} "
        f"best_epoch={result['best_epoch']} "
        f"({args.select_metric}={result['best_metric']:.4f})"
    )
    print(
        f"  auroc={best['auroc']:.4f} accuracy={best['accuracy']:.4f} "
        f"f1={best['f1']:.4f} positive_rate={best['positive_rate']:.4f}"
    )
    print(f"train_losses={[round(x, 4) for x in result['train_losses']]}")
    print(f"val_losses={[round(x, 4) for x in result['val_losses']]}")


def main() -> None:
    """Entry point for ``python -m nspe.train.cli``."""
    args = build_parser().parse_args()
    result = train_one(args)
    _print_result(args, result)

    if args.metrics_out is not None:
        with open(args.metrics_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.metrics_out}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
