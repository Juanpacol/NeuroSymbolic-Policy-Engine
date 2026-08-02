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
from nspe.calibration import VerdictCalibrator
from nspe.engine import PolicyEngine
from nspe.eval.diagnostics import predicate_stats, signature_distribution
from nspe.eval.hateful_memes import compute_h1, compute_h3, sample_explanations
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
from nspe.train.loop import load_trainable_state_dict

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
    clip_model: str = "ViT-L-14",
    clip_pretrained: str = "openai",
    hidden_dim: int = 256,
    cache_dir: str | None = None,
    threshold: float | None = None,
    learnable_confidence: bool = False,
    aggregate: str = "tconorm",
    pmean_p: float = 2.0,
) -> dict[str, Any]:
    """Runs the reasoner and baseline over one split and computes H1/H3.

    Args:
        policy_path: path to the policy YAML used to train the reasoner.
        reasoner_checkpoint: path to the trained extractor state dict.
        baseline_checkpoint: path to the trained baseline state dict.
        split: dataset split (``"validation"`` or ``"test"``).
        device: ``"cpu"``, ``"mps"``, or ``"cuda"``.
        batch_size: evaluation batch size.
        clip_model: ``open_clip`` architecture the checkpoints were
            trained with. Must match, or ``load_state_dict`` will fail
            on the head shapes.
        clip_pretrained: ``open_clip`` pretrained tag, likewise.
        hidden_dim: shared trunk width the checkpoints were trained
            with. Must match, likewise.
        cache_dir: directory of precomputed CLIP embeddings, reusing the
            training cache so evaluation does not re-encode the split.
        threshold: verdict threshold. ``None`` fits one per model on
            this split, which is only legitimate for validation -- pass
            the validation-fitted value when reporting test.
        learnable_confidence: whether the checkpoint was trained with
            trainable rule confidences.
        aggregate: how rules sharing a head combine. Must match what the
            checkpoint was trained with: unlike the weights, this is
            plain Python state and is *not* carried in the state dict,
            so a mismatch silently evaluates the model under an
            aggregation it never saw.
        pmean_p: power for ``aggregate="pmean"``, likewise not carried
            in the state dict.

    Returns:
        A dict with ``dataset``, ``h1_consistency``, and
        ``h3_explainability`` entries, ready to merge into the CLI's
        JSON output.
    """
    from nspe.data.hateful_memes import HatefulMemesDataset

    policy = load_policy(policy_path)
    extractor = NeuroSymbolicLayer.from_policy(
        policy,
        model_name=clip_model,
        pretrained=clip_pretrained,
        hidden_dim=hidden_dim,
        # The checkpoint carries the trained zero-shot buffer; re-encoding
        # descriptions here would just be overwritten by load_state_dict.
        init_from_descriptions=False,
    )
    reasoner = PolicyKGReasoner(
        policy,
        store_trace=False,
        learnable_confidence=learnable_confidence,
        aggregate=aggregate,
        pmean_p=pmean_p,
    )
    # The training CLI checkpoints the whole PolicyEngine, so the state
    # dict is keyed "extractor.*"/"reasoner.*"/"calibrator.*"; load it
    # as one. Both arms are built with a calibrator so the shapes match
    # what training saved.
    engine = PolicyEngine(extractor, reasoner, calibrator=VerdictCalibrator())
    engine = engine.to(device)
    load_trainable_state_dict(
        engine, torch.load(reasoner_checkpoint, weights_only=True)
    )
    engine.eval()

    baseline = NeuralBaselineClassifier(
        model_name=clip_model,
        pretrained=clip_pretrained,
        num_predicates=len(policy.predicate_names("base")),
        hidden_dim=hidden_dim,
        calibrator=VerdictCalibrator(),
    ).to(device)
    load_trainable_state_dict(
        baseline, torch.load(baseline_checkpoint, weights_only=True)
    )
    baseline.eval()

    if cache_dir is not None:
        path = cache_path(cache_dir, split, clip_model, clip_pretrained)
        if not path.exists():
            print(f"Encoding {split} split -> {path} (one-time)")
            precompute_embeddings(
                extractor,
                HatefulMemesDataset(split=split, transform=extractor.preprocess),
                path,
                batch_size=batch_size,
                device=device,
                model_name=clip_model,
                pretrained=clip_pretrained,
            )
        loader = DataLoader(
            EmbeddingDataset(
                path, expect_model=clip_model, expect_pretrained=clip_pretrained
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_embeddings,
        )
        num_examples = len(loader.dataset)  # type: ignore[arg-type]
    else:
        dataset = HatefulMemesDataset(split=split, transform=extractor.preprocess)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_hateful_memes,
        )
        num_examples = len(dataset)

    all_mu0, all_reasoner_verdict, all_baseline_verdict, all_labels = [], [], [], []
    for inputs, texts, labels in loader:
        inputs = inputs.to(device)
        fused = inputs if cache_dir is not None else extractor.encode(inputs, texts)
        mu0 = extractor.forward_embedded(fused)
        reasoner_out = engine.reasoner(mu0)
        verdict = reasoner_out.verdicts[_VERDICT_NAME]
        if engine.calibrator is not None:
            verdict = engine.calibrator(verdict)

        all_mu0.append(mu0.cpu())
        all_reasoner_verdict.append(verdict.cpu())
        all_baseline_verdict.append(baseline.forward_embedded(fused).cpu())
        all_labels.append(labels)

    mu0 = torch.cat(all_mu0)
    reasoner_verdict = torch.cat(all_reasoner_verdict)
    baseline_verdict = torch.cat(all_baseline_verdict)
    labels = torch.cat(all_labels)

    h1 = compute_h1(mu0, reasoner_verdict, baseline_verdict)
    h3 = compute_h3(reasoner_verdict, baseline_verdict, labels, threshold=threshold)

    predicate_names = policy.predicate_names("base")
    h1["predicate_stats"] = predicate_stats(mu0, predicate_names)
    h1["signature_distribution"] = signature_distribution(
        mu0, predicate_names, top_k=10
    )

    reasoner_pred = reasoner_verdict >= h3["reasoner"]["threshold"]
    baseline_pred = baseline_verdict >= h3["baseline"]["threshold"]
    disagreements = (reasoner_pred != baseline_pred).nonzero(as_tuple=True)[0]
    sample_indices = disagreements[:5].tolist()
    h3["sample_explanations"] = sample_explanations(
        policy, reasoner, mu0, sample_indices, target=_VERDICT_NAME
    )

    return {
        "dataset": {
            "name": "neuralcatcher/hateful_memes",
            "split": split,
            "num_examples": num_examples,
        },
        # Recorded because none of it lives in the checkpoint, so a
        # results file is otherwise ambiguous about what was evaluated.
        "reasoner_config": {
            "learnable_confidence": learnable_confidence,
            "aggregate": aggregate,
            "pmean_p": pmean_p,
            "clip_model": clip_model,
            "clip_pretrained": clip_pretrained,
            "hidden_dim": hidden_dim,
        },
        "h1_consistency": h1,
        "h3_explainability": h3,
    }


def _print_markdown(result: dict[str, Any]) -> None:
    h1 = result["h1_consistency"]
    print("### H1 consistency")
    print("| model | inconsistency | null | adjusted | positive_rate | classes |")
    print("|---|---|---|---|---|---|")
    for name in ("reasoner", "baseline"):
        r = h1[name]
        adjusted = (
            "n/a (degenerate)"
            if r["degenerate"]
            else f"{r['adjusted_consistency']:.4f}"
        )
        print(
            f"| {name} | {r['inconsistency_rate']:.4f} "
            f"| {r['null_inconsistency']:.4f} | {adjusted} "
            f"| {r['positive_rate']:.4f} | {r['num_classes']} |"
        )
    print(
        "\nRaw inconsistency is not comparable across models on its own: a "
        "model predicting one class scores a perfect 0. Read "
        "`adjusted` (1 = perfect given the marginal, 0 = chance) "
        "alongside `positive_rate`."
    )

    print("\n### Predicate activation")
    print("| predicate | activation_rate | std | max_abs_corr |")
    print("|---|---|---|---|")
    for name, stats in h1["predicate_stats"].items():
        print(
            f"| {name} | {stats['activation_rate']:.4f} | {stats['std']:.4f} "
            f"| {stats['max_abs_correlation']:.4f} |"
        )
    print(
        f"\nsignature_entropy = {h1['reasoner']['signature_entropy']:.3f} bits "
        f"over {h1['reasoner']['num_classes']} classes"
    )

    h3 = result["h3_explainability"]
    print(f"\n### H3 accuracy (threshold {h3['threshold_source']})")
    print("| model | auroc | accuracy | f1 | precision | recall | threshold |")
    print("|---|---|---|---|---|---|---|")
    for name in ("reasoner", "baseline"):
        m = h3[name]
        print(
            f"| {name} | {m['auroc']:.4f} | {m['accuracy']:.4f} | {m['f1']:.4f} "
            f"| {m['precision']:.4f} | {m['recall']:.4f} | {m['threshold']:.4f} |"
        )
    print(
        f"\nmajority-class accuracy = {h3['majority_class_accuracy']:.4f} "
        "(any model below this has learned nothing usable)"
    )
    print(
        f"\nauroc_gap (reasoner - baseline) = {h3['auroc_gap']:.4f}, "
        f"accuracy_gap = {h3['accuracy_gap']:.4f}, f1_gap = {h3['f1_gap']:.4f}"
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
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--learnable-confidence",
        action="store_true",
        help="Set if the checkpoint was trained with --learnable-confidence.",
    )
    parser.add_argument(
        "--aggregate",
        default="tconorm",
        choices=["tconorm", "pmean"],
        help="Must match the checkpoint's training setting: aggregation "
        "is not carried in the state dict, so a mismatch silently "
        "evaluates the model under an aggregation it never saw.",
    )
    parser.add_argument("--pmean-p", type=float, default=2.0)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Reuse the training embedding cache instead of re-encoding.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Verdict threshold. Omit to fit one per model on this "
        "split -- legitimate for validation only. When reporting test, "
        "pass the value fitted on validation.",
    )
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
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        hidden_dim=args.hidden_dim,
        cache_dir=args.cache_dir,
        threshold=args.threshold,
        learnable_confidence=args.learnable_confidence,
        aggregate=args.aggregate,
        pmean_p=args.pmean_p,
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
