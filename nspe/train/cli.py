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
"""

from __future__ import annotations

import argparse

from torch import Tensor
from torch.utils.data import DataLoader

from nspe.baselines.neural_classifier import NeuralBaselineClassifier
from nspe.engine import PolicyEngine
from nspe.extractor import NeuroSymbolicLayer
from nspe.policy.loader import load_policy
from nspe.reasoner import PolicyKGReasoner
from nspe.train.dataset import collate_hateful_memes
from nspe.train.loop import train_model

_VERDICT_NAME = "hateful"


def _reasoner_forward(model: PolicyEngine, images: Tensor, texts: list[str]) -> Tensor:
    return model(images, texts).verdicts[_VERDICT_NAME]


def _baseline_forward(
    model: NeuralBaselineClassifier, images: Tensor, texts: list[str]
) -> Tensor:
    return model(images, texts)


def _build_model(model_kind: str, policy_path: str):
    from nspe.data.hateful_memes import HatefulMemesDataset

    if model_kind == "reasoner":
        policy = load_policy(policy_path)
        extractor = NeuroSymbolicLayer.from_policy(policy)
        reasoner = PolicyKGReasoner(policy, store_trace=False)
        model = PolicyEngine(extractor, reasoner)
        preprocess = extractor.preprocess
        forward_fn = _reasoner_forward
    else:
        model = NeuralBaselineClassifier()
        preprocess = model.preprocess
        forward_fn = _baseline_forward

    train_ds = HatefulMemesDataset(split="train", transform=preprocess)
    val_ds = HatefulMemesDataset(split="validation", transform=preprocess)
    return model, forward_fn, train_ds, val_ds


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
    args = parser.parse_args()

    model, forward_fn, train_ds, val_ds = _build_model(args.model, args.policy)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_hateful_memes,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_hateful_memes,
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
    )

    print(f"model={args.model} best_val_loss={result['best_val_loss']:.4f}")
    print(f"train_losses={[round(x, 4) for x in result['train_losses']]}")
    print(f"val_losses={[round(x, 4) for x in result['val_losses']]}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
