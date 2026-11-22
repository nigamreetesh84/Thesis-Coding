"""CLI for the federated-learning experiment (RQ5).

    # 1. data only (CPU, seconds): writes results_fl/data/*.jsonl and a summary
    python -m fl.run_fl data

    # 2. full experiment on a GPU (Colab T4: about 1.5-2.5 h with the defaults)
    python -m fl.run_fl train --out-dir results_fl/run1

    # quicker run (0.5B model, fewer steps) to check everything works first
    python -m fl.run_fl train --preset quick --out-dir results_fl/quick

    # 3. tables + exact McNemar from a finished (or partly finished) run
    python -m fl.run_fl report --out-dir results_fl/run1

The run is resumable: arms already in results.json are skipped, so after a Colab
disconnect just run the same command again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fl.data_gen import build_datasets, save, summary
from fl.train import EVAL_SETS, FLConfig, per_card_mcnemar, run_experiment

PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "quick": {"model_name": "Qwen/Qwen2.5-0.5B-Instruct", "rounds": 3, "local_steps": 8, "eval_limit": 6},
}


def report(out_dir: Path) -> str:
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    arms = results["arms"]
    order = [a for a in ("zero_shot", "local/SBI", "local/HDFC", "local/AXIS", "centralised", "fedavg", "fedavg_lobo")
             if a in arms]
    set_names = sorted({s for a in order for s in arms[a]["sets"]},
                       key=lambda s: (EVAL_SETS.index(s.split("/")[0]), s))
    lines = ["# RQ5 – federated rule extractor: field accuracy (correct fields / total)", "",
             f"Model: {results['config']['model_name']} · rounds {results['config']['rounds']} · "
             f"local steps {results['config']['local_steps']} · LoRA r={results['config']['lora_r']}", ""]
    lines.append("| Test set | " + " | ".join(order) + " |")
    lines.append("|---|" + "---|" * len(order))
    for s in set_names:
        cells = []
        for a in order:
            m = arms[a]["sets"].get(s)
            cells.append(f"{m['fields_correct']}/{m['fields_total']} ({m['field_accuracy']:.2f})" if m else "–")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    summ = results.get("comparisons", {}).get("summary", {})
    if summ:
        lines += ["", "| Pooled | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
        for fam in ("all_unseen", "real_documents", "synthetic"):
            cells = [f"{summ[a][fam]['fields_correct']}/{summ[a][fam]['fields_total']} ({summ[a][fam]['field_accuracy']:.2f})"
                     for a in order if a in summ]
            lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    pairs = results.get("comparisons", {}).get("pairs", [])
    if pairs:
        stats = per_card_mcnemar(out_dir, pairs)
        lines += ["", "Exact McNemar on cards with all five fields correct (pooled unseen sets; injection set excluded):", ""]
        lines += [f"- {k}: b={v['b_first_only']} c={v['c_second_only']} p={v['p_value']:.4g} (n={v['n']})"
                  for k, v in stats.items()]
        results["per_card_mcnemar"] = stats
    inj = {a: arms[a]["sets"].get("injection_axis") for a in order}
    if any(inj.values()):
        lines += ["", "Prompt-injection file (Axis page with an injected instruction): field accuracy",
                  ", ".join(f"{a} {m['fields_correct']}/{m['fields_total']}" for a, m in inj.items() if m)]
    hall = {a: sum(m["hallucinated_unstated_thresholds"] for m in arms[a]["sets"].values()) for a in order}
    lines += ["", "Invented thresholds where the document states none (income/CIBIL): "
              + ", ".join(f"{a} {v}" for a, v in hall.items())]
    if "fedavg" in arms and "bytes_per_upload" in arms["fedavg"]:
        lines += ["", f"Data sent per client per round: {arms['fedavg']['bytes_per_upload'] / 1e6:.1f} MB of adapter weights; "
                  "no training example leaves its bank."]
    text = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Federated rule-extractor experiment (RQ5)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--out-dir", type=Path, default=Path("results_fl/data"))
    t = sub.add_parser("train")
    t.add_argument("--out-dir", type=Path, default=Path("results_fl/run1"))
    t.add_argument("--preset", choices=list(PRESETS), default="default")
    t.add_argument("--model-name")
    t.add_argument("--rounds", type=int)
    t.add_argument("--local-steps", type=int)
    t.add_argument("--arms", nargs="*")
    t.add_argument("--no-4bit", action="store_true")
    t.add_argument("--no-portal-corrections", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--out-dir", type=Path, default=Path("results_fl/run1"))
    args = ap.parse_args()

    if args.cmd == "data":
        ds = build_datasets()
        save(ds, args.out_dir)
        print(json.dumps(summary(ds), indent=1))
    elif args.cmd == "train":
        cfg = FLConfig(**PRESETS[args.preset])
        if args.model_name:
            cfg.model_name = args.model_name
        if args.rounds:
            cfg.rounds = args.rounds
        if args.local_steps:
            cfg.local_steps = args.local_steps
        if args.arms:
            cfg.arms = tuple(args.arms)
        if args.no_4bit:
            cfg.quantize_4bit = False
        ds = build_datasets(portal_dir=None if args.no_portal_corrections else Path("data/portal"))
        save(ds, args.out_dir / "data")
        run_experiment(cfg, ds, args.out_dir)
        print(report(args.out_dir))
    else:
        print(report(args.out_dir))


if __name__ == "__main__":
    main()
