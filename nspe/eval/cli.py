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
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from nspe.baselines.neural_classifier import NeuralBaselineClassifier
from nspe.calibration import VerdictCalibrator
from nspe.engine import PolicyEngine
from nspe.eval.diagnostics import predicate_stats, signature_distribution
from nspe.eval.hateful_memes import (
    compute_calibration,
    compute_h1,
    compute_h3,
    sample_explanations,
)
from nspe.extractor import NeuroSymbolicLayer
from nspe.policy.loader import load_policy
from nspe.policy.schema import Policy
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
_NUM_EXPLANATIONS = 5


def _spread_sample(indices: torch.Tensor, count: int) -> list[int]:
    """Picks ``count`` indices spread across ``indices``, not the head.

    This dataset's rows are ordered by label, so taking the first few
    disagreements draws every sampled explanation from a single class --
    which makes the published audit chains unrepresentative of the
    split in exactly the dimension they are meant to illustrate.

    Args:
        indices: candidate row indices, ascending.
        count: how many to select.

    Returns:
        Evenly spaced indices, or all of them if there are fewer than
        ``count``.
    """
    if indices.numel() <= count:
        return indices.tolist()
    positions = torch.linspace(0, indices.numel() - 1, count).long()
    return indices[positions].tolist()


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


_MU0_DECIMALS = 6


def _write_mu0_dump(
    path: str,
    mu0: torch.Tensor,
    labels: torch.Tensor,
    policy: Policy,
    predicate_names: tuple[str, ...],
    device: str,
) -> None:
    """Writes base predicate degrees for policy-varying analyses.

    ``predicate_names`` and ``policy_fingerprint`` are what make the
    columns interpretable and the dump attributable; without them a
    consumer cannot tell a column reordering from a real difference.
    :func:`nspe.eval.wiring_sweep.load_mu0_dump` refuses a dump whose
    either field disagrees with the policy it was handed.

    Args:
        path: destination JSON file.
        mu0: base predicate truth degrees, shape ``(n, P)``.
        labels: binary ground truth, shape ``(n,)``.
        policy: the policy the reasoner ran.
        predicate_names: base predicate names, in column order.
        device: the device the run used, recorded for provenance.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "policy_name": policy.name,
                "policy_fingerprint": PolicyKGReasoner(policy).rule_tensor.fingerprint,
                "predicate_names": list(predicate_names),
                "environment": _environment_metadata(device),
                "mu0": [[round(v, _MU0_DECIMALS) for v in row] for row in mu0.tolist()],
                "labels": [int(v) for v in labels.tolist()],
            }
        )
    )
    print(f"wrote mu0 dump -> {out_path}")


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
    threshold: float | tuple[float, float] | None = None,
    learnable_confidence: bool = False,
    aggregate: str = "tconorm",
    pmean_p: float = 2.0,
    dump_mu0: str | None = None,
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
            this split, which is only legitimate for validation; a float
            shares one point across both arms; a ``(reasoner,
            baseline)`` pair gives each its own, which is what a test
            run needs. See :func:`resolve_thresholds`.
        learnable_confidence: whether the checkpoint was trained with
            trainable rule confidences.
        aggregate: how rules sharing a head combine. Must match what the
            checkpoint was trained with: unlike the weights, this is
            plain Python state and is *not* carried in the state dict,
            so a mismatch silently evaluates the model under an
            aggregation it never saw.
        pmean_p: power for ``aggregate="pmean"``, likewise not carried
            in the state dict.
        dump_mu0: if given, a path to write the raw base predicate
            degrees and labels to, for analyses that vary the policy
            over a fixed predicate layer -- see
            :mod:`nspe.eval.wiring_sweep`.

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

    loader: DataLoader[Any]
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

    all_mu0, all_reasoner_verdict, all_labels = [], [], []
    all_reasoner_raw, all_baseline_raw = [], []
    # Collect the baseline's pre-calibration verdict by switching its
    # calibrator off for the pass, then applying it once afterwards --
    # the map is elementwise with no batch state, so that is exactly
    # equivalent and avoids a second forward pass.
    if baseline.calibrator is not None:
        baseline.calibrator.enabled = False

    for inputs, texts, labels in loader:
        inputs = inputs.to(device)
        fused = inputs if cache_dir is not None else extractor.encode(inputs, texts)
        mu0 = extractor.forward_embedded(fused)
        reasoner_out = engine.reasoner(mu0)
        raw_verdict = reasoner_out.verdicts[_VERDICT_NAME]
        verdict = raw_verdict
        if engine.calibrator is not None:
            verdict = engine.calibrator(verdict)

        all_mu0.append(mu0.cpu())
        all_reasoner_raw.append(raw_verdict.cpu())
        all_reasoner_verdict.append(verdict.cpu())
        all_baseline_raw.append(baseline.forward_embedded(fused).cpu())
        all_labels.append(labels)

    mu0 = torch.cat(all_mu0)
    reasoner_raw = torch.cat(all_reasoner_raw)
    reasoner_verdict = torch.cat(all_reasoner_verdict)
    baseline_raw = torch.cat(all_baseline_raw)
    labels = torch.cat(all_labels)

    if baseline.calibrator is not None:
        baseline.calibrator.enabled = True
        baseline_verdict = baseline.calibrator(baseline_raw.to(device)).cpu()
    else:
        baseline_verdict = baseline_raw

    # h1/h3 keep consuming the calibrated verdicts; feeding them the raw
    # ones would move every already-published number.
    h1 = compute_h1(mu0, reasoner_verdict, baseline_verdict)
    h3 = compute_h3(reasoner_verdict, baseline_verdict, labels, threshold=threshold)
    calibration = compute_calibration(
        reasoner_raw, reasoner_verdict, baseline_raw, baseline_verdict, labels
    )

    predicate_names = policy.predicate_names("base")
    h1["predicate_stats"] = predicate_stats(mu0, predicate_names, labels=labels)
    h1["signature_distribution"] = signature_distribution(
        mu0, predicate_names, top_k=10
    )

    if dump_mu0 is not None:
        _write_mu0_dump(dump_mu0, mu0, labels, policy, predicate_names, device)

    reasoner_pred = reasoner_verdict >= h3["reasoner"]["threshold"]
    baseline_pred = baseline_verdict >= h3["baseline"]["threshold"]
    disagreements = (reasoner_pred != baseline_pred).nonzero(as_tuple=True)[0]
    sample_indices = _spread_sample(disagreements, _NUM_EXPLANATIONS)
    h3["sample_explanations"] = sample_explanations(
        policy, reasoner, mu0, sample_indices, target=_VERDICT_NAME
    )
    # The chains alone cannot say whether the reasoner was right. These
    # are already in scope, so attaching them costs nothing and turns a
    # sampled explanation into an auditable one.
    for entry in h3["sample_explanations"]:
        case = entry["case_index"]
        entry["label"] = float(labels[case].item())
        entry["reasoner_verdict"] = float(reasoner_verdict[case].item())
        entry["baseline_verdict"] = float(baseline_verdict[case].item())
        entry["reasoner_pred"] = bool(reasoner_pred[case].item())
        entry["baseline_pred"] = bool(baseline_pred[case].item())

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
        # Top-level rather than nested inside h3: nspe/eval/aggregate.py
        # detects artifact shape by the presence of "h3_explainability"
        # and reads its arms positionally, so a new sub-block there
        # would risk colliding with that.
        "calibration": calibration,
    }


def resolve_thresholds(
    args: argparse.Namespace,
) -> float | tuple[float, float] | None:
    """Resolves this run's operating point from the CLI flags.

    ``--thresholds-from`` exists because the threshold has to belong to
    the checkpoint it is applied to. Checkpoints are gitignored, so a
    test run is necessarily evaluated against weights retrained in that
    session, and the calibrator's operating point is not stable across
    retrains even at a fixed seed -- reusing a threshold from an older
    artifact would be a silent protocol error. Reading it out of the
    validation results produced by *these* checkpoints removes both the
    transcription step and that hazard, and lets the backbone be
    cross-checked against the artifact that fitted it.

    Args:
        args: a namespace from the evaluation CLI's parser.

    Returns:
        ``None`` to fit per arm on the evaluated split, a float for one
        shared threshold, or a ``(reasoner, baseline)`` pair.

    Raises:
        ValueError: if the split is ``"test"`` and no threshold source
            was given, if only one per-arm flag was passed, or if the
            source artifact was not itself fitted or came from a
            different backbone.
    """
    if args.reasoner_threshold is not None or args.baseline_threshold is not None:
        if args.reasoner_threshold is None or args.baseline_threshold is None:
            raise ValueError(
                "--reasoner-threshold and --baseline-threshold must be "
                "given together; they are one operating point per arm"
            )
        return (args.reasoner_threshold, args.baseline_threshold)

    if args.thresholds_from is not None:
        source = json.loads(Path(args.thresholds_from).read_text())
        h3 = source["h3_explainability"]
        if h3.get("threshold_source") != "fitted":
            raise ValueError(
                f"{args.thresholds_from} has threshold_source="
                f"{h3.get('threshold_source')!r}; chaining a threshold that "
                "was itself provided says nothing about where it came from"
            )

        recorded = source.get("reasoner_config", {}).get("clip_model")
        if recorded is not None and recorded != args.clip_model:
            raise ValueError(
                f"{args.thresholds_from} was produced with clip_model="
                f"{recorded!r} but this run uses {args.clip_model!r}"
            )
        if source.get("policy_fingerprint") not in (None, _fingerprint(args.policy)):
            print(
                f"WARNING: {args.thresholds_from} has a different "
                "policy_fingerprint than this run's policy"
            )
        # h3_explainability, never h1_consistency: the latter's rates are
        # taken at the H1 verdict threshold of 0.5, a different point.
        return (h3["reasoner"]["threshold"], h3["baseline"]["threshold"])

    if args.split == "test":
        raise ValueError(
            "refusing to fit a threshold on the test split: pass "
            "--thresholds-from <validation results.json> (or explicit "
            "--reasoner-threshold/--baseline-threshold) so the operating "
            "point comes from validation"
        )
    # args.threshold comes from argparse.Namespace, typed Any.
    return cast("float | tuple[float, float] | None", args.threshold)


def _fingerprint(policy_path: str) -> str:
    return PolicyKGReasoner(load_policy(policy_path)).rule_tensor.fingerprint


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
    point = parser.add_mutually_exclusive_group()
    point.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="One verdict threshold shared by both arms. Omit to fit "
        "per arm on this split -- legitimate for validation only.",
    )
    point.add_argument(
        "--thresholds-from",
        type=str,
        default=None,
        help="Read each arm's threshold from a validation results JSON "
        "produced by these same checkpoints. Required for --split test.",
    )
    parser.add_argument(
        "--reasoner-threshold",
        type=float,
        default=None,
        help="Explicit per-arm operating point; use with "
        "--baseline-threshold when there is no validation artifact.",
    )
    parser.add_argument("--baseline-threshold", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--dump-mu0",
        type=str,
        default=None,
        help="write base predicate degrees and labels here, for nspe.eval.wiring_sweep",
    )
    args = parser.parse_args()

    thresholds = resolve_thresholds(args)
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
        threshold=thresholds,
        learnable_confidence=args.learnable_confidence,
        aggregate=args.aggregate,
        pmean_p=args.pmean_p,
        dump_mu0=args.dump_mu0,
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
