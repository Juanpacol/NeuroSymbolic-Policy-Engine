"""H1/H3 evaluation CLI: reasoner vs. neural baseline on Hateful Memes.

Usage:
    python -m nspe.eval.cli --policy nspe/policies/hateful_memes.yaml \
        --reasoner-checkpoint checkpoints/reasoner.pt \
        --baseline-checkpoint checkpoints/baseline.pt \
        --split validation --device cuda --out eval_results/h1_h3.json

Loads both trained checkpoints, runs them over the requested split, and
reports:
  - H1 (consistency): paired ConsistencyChecker reports, both grouped by
    the reasoner's own mu0 (see nspe.eval.hateful_memes.compute_h1).
  - H3 (explainability without accuracy loss): accuracy/F1 for each
    model against the real label, plus sample audit-chain explanations.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from nspe.baselines.neural_classifier import NeuralBaselineClassifier
from nspe.eval.hateful_memes import compute_h1, compute_h3, sample_explanations
from nspe.extractor import NeuroSymbolicLayer
from nspe.policy.loader import load_policy
from nspe.reasoner import PolicyKGReasoner
from nspe.train.dataset import collate_hateful_memes

_VERDICT_NAME = "hateful"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _environment_metadata(device: str) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "git_commit": _git_commit(),
    }


@torch.no_grad()
def run_eval(
    policy_path: str,
    reasoner_checkpoint: str,
    baseline_checkpoint: str,
    split: str,
    device: str,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Runs the reasoner and baseline over one split and computes H1/H3.

    Args:
        policy_path: path to the policy YAML used to train the reasoner.
        reasoner_checkpoint: path to the trained extractor state dict.
        baseline_checkpoint: path to the trained baseline state dict.
        split: dataset split (``"validation"`` or ``"test"``).
        device: ``"cpu"``, ``"mps"``, or ``"cuda"``.
        batch_size: evaluation batch size.

    Returns:
        A dict with ``dataset``, ``h1_consistency``, and
        ``h3_explainability`` entries, ready to merge into the CLI's
        JSON output.
    """
    from nspe.data.hateful_memes import HatefulMemesDataset

    policy = load_policy(policy_path)
    extractor = NeuroSymbolicLayer.from_policy(policy).to(device)
    extractor.load_state_dict(torch.load(reasoner_checkpoint, weights_only=True))
    extractor.eval()
    reasoner = PolicyKGReasoner(policy, store_trace=False).to(device)

    baseline = NeuralBaselineClassifier().to(device)
    baseline.load_state_dict(torch.load(baseline_checkpoint, weights_only=True))
    baseline.eval()

    dataset = HatefulMemesDataset(split=split, transform=extractor.preprocess)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_hateful_memes
    )

    all_mu0, all_reasoner_verdict, all_baseline_verdict, all_labels = [], [], [], []
    for images, texts, labels in loader:
        images = images.to(device)
        mu0 = extractor(images, texts)
        reasoner_out = reasoner(mu0)
        baseline_verdict = baseline(images, texts)

        all_mu0.append(mu0.cpu())
        all_reasoner_verdict.append(reasoner_out.verdicts[_VERDICT_NAME].cpu())
        all_baseline_verdict.append(baseline_verdict.cpu())
        all_labels.append(labels)

    mu0 = torch.cat(all_mu0)
    reasoner_verdict = torch.cat(all_reasoner_verdict)
    baseline_verdict = torch.cat(all_baseline_verdict)
    labels = torch.cat(all_labels)

    h1 = compute_h1(mu0, reasoner_verdict, baseline_verdict)
    h3 = compute_h3(reasoner_verdict, baseline_verdict, labels)

    reasoner_pred = reasoner_verdict >= 0.5
    baseline_pred = baseline_verdict >= 0.5
    disagreements = (reasoner_pred != baseline_pred).nonzero(as_tuple=True)[0]
    sample_indices = disagreements[:5].tolist()
    h3["sample_explanations"] = sample_explanations(
        policy, reasoner, mu0, sample_indices, target=_VERDICT_NAME
    )

    return {
        "dataset": {
            "name": "neuralcatcher/hateful_memes",
            "split": split,
            "num_examples": len(dataset),
        },
        "h1_consistency": h1,
        "h3_explainability": h3,
    }


def _print_markdown(result: dict[str, Any]) -> None:
    h1 = result["h1_consistency"]
    print("| model | inconsistency_rate | purity | num_classes |")
    print("|---|---|---|---|")
    for name in ("reasoner", "baseline"):
        r = h1[name]
        print(
            f"| {name} | {r['inconsistency_rate']:.4f} | {r['purity']:.4f} "
            f"| {r['num_classes']} |"
        )

    h3 = result["h3_explainability"]
    print("\n| model | accuracy | f1 |")
    print("|---|---|---|")
    for name in ("reasoner", "baseline"):
        m = h3[name]
        print(f"| {name} | {m['accuracy']:.4f} | {m['f1']:.4f} |")
    print(
        f"\naccuracy_gap (reasoner - baseline) = {h3['accuracy_gap']:.4f}, "
        f"f1_gap = {h3['f1_gap']:.4f}"
    )


def main() -> None:
    """Entry point for ``python -m nspe.eval.cli``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="nspe/policies/hateful_memes.yaml")
    parser.add_argument("--reasoner-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    eval_result = run_eval(
        args.policy,
        args.reasoner_checkpoint,
        args.baseline_checkpoint,
        args.split,
        args.device,
        args.batch_size,
    )
    _print_markdown(eval_result)

    result = {
        "environment": _environment_metadata(args.device),
        "policy_name": policy.name,
        "policy_fingerprint": PolicyKGReasoner(policy).rule_tensor.fingerprint,
        "checkpoints": {
            "reasoner": args.reasoner_checkpoint,
            "baseline": args.baseline_checkpoint,
        },
        **eval_result,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
