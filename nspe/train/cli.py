"""Training CLI: fits either the reasoner path or the neural baseline.

Usage:
    python -m nspe.train.cli --model reasoner \
        --policy nspe/policies/hateful_memes.yaml --device cuda \
        --out checkpoints/reasoner.pt
    python -m nspe.train.cli --model baseline --device cuda \
        --out checkpoints/baseline.pt

Both are trained identically: BCE on the "hateful" verdict against
Hateful Memes' real binary label. For --model reasoner, gradients flow
through the differentiable PolicyKGReasoner into the extractor's
predicate heads; for --model baseline, only the single linear head is
trained. CLIP stays frozen in both cases.

Because CLIP is frozen, --cache-dir avoids re-encoding the split on
every epoch: pass a directory and the first run encodes each split once
to disk, while that run and every later one train off those cached
embeddings. On a free GPU session this is the difference between an
epoch dominated by image downloads and an epoch that takes seconds.
Cache files are per (split, CLIP architecture); delete the directory if
you change the backbone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from nspe.baselines.neural_classifier import NeuralBaselineClassifier
from nspe.engine import PolicyEngine
from nspe.extractor import NeuroSymbolicLayer
from nspe.policy.loader import load_policy
from nspe.reasoner import PolicyKGReasoner
from nspe.train.cache import (
    EmbeddingDataset,
    collate_embeddings,
    precompute_embeddings,
)
from nspe.train.dataset import collate_hateful_memes
from nspe.train.loop import train_model

_VERDICT_NAME = "hateful"


def _reasoner_forward(model: PolicyEngine, images: Tensor, texts: list[str]) -> Tensor:
    return model(images, texts).verdicts[_VERDICT_NAME]


def _reasoner_forward_embedded(model: PolicyEngine, fused: Tensor, _: None) -> Tensor:
    return model.forward_embedded(fused).verdicts[_VERDICT_NAME]


def _baseline_forward(
    model: NeuralBaselineClassifier, images: Tensor, texts: list[str]
) -> Tensor:
    return model(images, texts)


def _baseline_forward_embedded(
    model: NeuralBaselineClassifier, fused: Tensor, _: None
) -> Tensor:
    return model.forward_embedded(fused)


def _build_model(model_kind: str, policy_path: str):
    if model_kind == "reasoner":
        policy = load_policy(policy_path)
        extractor = NeuroSymbolicLayer.from_policy(policy)
        reasoner = PolicyKGReasoner(policy, store_trace=False)
        return PolicyEngine(extractor, reasoner), extractor.preprocess
    model = NeuralBaselineClassifier()
    return model, model.preprocess


def _encoder_of(model: nn.Module) -> nn.Module:
    return model.extractor if isinstance(model, PolicyEngine) else model


def _cached_loaders(
    model: nn.Module,
    preprocess,
    cache_dir: str,
    batch_size: int,
    device: str,
    limit_train: int | None,
    limit_val: int | None,
) -> tuple[DataLoader, DataLoader]:
    """Builds loaders over cached embeddings, encoding splits if needed."""
    encoder = _encoder_of(model)
    arch = f"{encoder.clip.visual.output_dim}d"
    loaders = []
    for split, limit in (("train", limit_train), ("validation", limit_val)):
        path = Path(cache_dir) / f"hateful_memes_{split}_{arch}.pt"
        if not path.exists():
            from nspe.data.hateful_memes import HatefulMemesDataset

            print(f"Encoding {split} split -> {path} (one-time)")
            precompute_embeddings(
                encoder,
                HatefulMemesDataset(split=split, transform=preprocess),
                path,
                batch_size=batch_size,
                device=device,
            )
        dataset = EmbeddingDataset(path, limit=limit)
        print(f"{split}: {len(dataset)} cached examples")
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                collate_fn=collate_embeddings,
            )
        )
    return loaders[0], loaders[1]


def _raw_loaders(
    preprocess,
    batch_size: int,
    limit_train: int | None,
    limit_val: int | None,
) -> tuple[DataLoader, DataLoader]:
    """Builds loaders that encode images on the fly, every epoch."""
    from nspe.data.hateful_memes import HatefulMemesDataset

    loaders = []
    for split, limit in (("train", limit_train), ("validation", limit_val)):
        dataset = HatefulMemesDataset(split=split, transform=preprocess)
        if limit is not None:
            dataset = Subset(dataset, range(min(limit, len(dataset))))
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=split == "train",
                collate_fn=collate_hateful_memes,
            )
        )
    return loaders[0], loaders[1]


def main() -> None:
    """Entry point for ``python -m nspe.train.cli``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["reasoner", "baseline"])
    parser.add_argument(
        "--policy",
        default="nspe/policies/hateful_memes.yaml",
        help="Path to a policy YAML file (ignored for --model baseline).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--out", type=str, required=True)
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
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Train on at most this many examples (debugging).",
    )
    parser.add_argument(
        "--limit-val",
        type=int,
        default=None,
        help="Validate on at most this many examples (debugging).",
    )
    args = parser.parse_args()

    model, preprocess = _build_model(args.model, args.policy)

    if args.cache_dir is not None:
        train_loader, val_loader = _cached_loaders(
            model,
            preprocess,
            args.cache_dir,
            args.batch_size,
            args.device,
            args.limit_train,
            args.limit_val,
        )
        forward_fn = (
            _reasoner_forward_embedded
            if args.model == "reasoner"
            else _baseline_forward_embedded
        )
    else:
        train_loader, val_loader = _raw_loaders(
            preprocess, args.batch_size, args.limit_train, args.limit_val
        )
        forward_fn = (
            _reasoner_forward if args.model == "reasoner" else _baseline_forward
        )

    result = train_model(
        model,
        forward_fn,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        checkpoint_path=args.out,
        resume_from=args.resume,
    )

    print(f"model={args.model} best_val_loss={result['best_val_loss']:.4f}")
    print(f"train_losses={[round(x, 4) for x in result['train_losses']]}")
    print(f"val_losses={[round(x, 4) for x in result['val_losses']]}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
