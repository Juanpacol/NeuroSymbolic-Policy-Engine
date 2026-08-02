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
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from nspe.baselines.neural_classifier import NeuralBaselineClassifier
from nspe.calibration import VerdictCalibrator
from nspe.engine import PolicyEngine
from nspe.extractor import NeuroSymbolicLayer
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


def _verdict_of(out: object) -> Tensor:
    """Prefers the calibrated verdict, falling back to the raw one."""
    calibrated = getattr(out, "calibrated", None)
    if calibrated is not None:
        return calibrated[_VERDICT_NAME]
    return out.verdicts[_VERDICT_NAME]  # type: ignore[attr-defined]


def _reasoner_forward(model: PolicyEngine, images: Tensor, texts: list[str]) -> Tensor:
    return _verdict_of(model(images, texts))


def _reasoner_forward_embedded(model: PolicyEngine, fused: Tensor, _: None) -> Tensor:
    return _verdict_of(model.forward_embedded(fused))


def _baseline_forward(
    model: NeuralBaselineClassifier, images: Tensor, texts: list[str]
) -> Tensor:
    return model(images, texts)


def _baseline_forward_embedded(
    model: NeuralBaselineClassifier, fused: Tensor, _: None
) -> Tensor:
    return model.forward_embedded(fused)


def _build_model(args: argparse.Namespace) -> tuple[nn.Module, object]:
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

    def aux(engine: nn.Module, inputs: Tensor, side: object) -> Tensor:
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


def _encoder_of(model: nn.Module) -> nn.Module:
    return model.extractor if isinstance(model, PolicyEngine) else model


def _cached_loaders(
    model: nn.Module, preprocess: object, args: argparse.Namespace
) -> tuple[DataLoader, DataLoader]:
    """Builds loaders over cached embeddings, encoding splits if needed."""
    encoder = _encoder_of(model)
    loaders = []
    for split, limit in (("train", args.limit_train), ("validation", args.limit_val)):
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
        print(f"{split}: {len(dataset)} cached examples")
        loaders.append(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=split == "train",
                collate_fn=collate_embeddings,
            )
        )
    return loaders[0], loaders[1]


def _raw_loaders(
    preprocess: object, args: argparse.Namespace
) -> tuple[DataLoader, DataLoader]:
    """Builds loaders that encode images on the fly, every epoch."""
    from nspe.data.hateful_memes import HatefulMemesDataset

    loaders = []
    for split, limit in (("train", args.limit_train), ("validation", args.limit_val)):
        dataset = HatefulMemesDataset(split=split, transform=preprocess)
        if limit is not None:
            dataset = Subset(dataset, range(min(limit, len(dataset))))
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
    forward_fn: object,
    loader: DataLoader,
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
        loader: the training loader.
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


def main() -> None:
    """Entry point for ``python -m nspe.train.cli``."""
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
    args = parser.parse_args()

    set_seed(args.seed)
    model, preprocess = _build_model(args)

    if args.cache_dir is not None:
        train_loader, val_loader = _cached_loaders(model, preprocess, args)
        forward_fn = (
            _reasoner_forward_embedded
            if args.model == "reasoner"
            else _baseline_forward_embedded
        )
    else:
        train_loader, val_loader = _raw_loaders(preprocess, args)
        forward_fn = (
            _reasoner_forward if args.model == "reasoner" else _baseline_forward
        )

    base_rate, pos_weight = _warm_start(model, forward_fn, train_loader, args.device)
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

    if args.metrics_out is not None:
        with open(args.metrics_out, "w") as f:
            json.dump({"args": vars(args), **result}, f, indent=2)
        print(f"Wrote {args.metrics_out}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
