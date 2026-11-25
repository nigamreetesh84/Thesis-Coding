"""Distributed run vs simulated run with identical inputs and settings.

    # 1. simulated FedAvg (Run-1 code path) on the SAME node folders and run config
    python -m fl.distributed.compare simulate --nodes deploy/nodes \
        --run-config fl/distributed/app/quick.toml --out results_fl/sim_quick

    # 2. side-by-side table: field accuracy per test set, per-card agreement,
    #    exact McNemar, and the largest difference between the two final adapters
    python -m fl.distributed.compare diff --sim results_fl/sim_quick --dist results_fl/distributed_quick

With lora-dropout = 0 both paths are deterministic and the final adapters agree
to float precision (tests/test_fl_distributed.py checks this); with dropout the
random draws differ between processes, so results are close, not identical.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from fl.data_gen import BANKS, Example
from fl.distributed.prepare import TRAIN_FILE, load_eval_sets, load_examples
from fl.distributed.settings import config_from_run
from fl.evaluate import mcnemar_cards

APP_DIR = Path(__file__).parent / "app"


def run_config(override: Path | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rc = dict(tomllib.loads((APP_DIR / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"])
    if override is not None:
        rc.update(tomllib.loads(override.read_text()))
    rc.update(extra or {})
    return rc


def node_data(nodes: Path, banks: tuple[str, ...] = BANKS) -> dict[str, list[Example]]:
    data = {f"train/{b}": load_examples(nodes / b / TRAIN_FILE) for b in banks}
    data.update(load_eval_sets(nodes / "server" / "eval"))
    return data


def simulate(nodes: Path, rc: dict[str, Any], out: Path) -> dict[str, Any]:
    from fl.train import run_experiment

    cfg = config_from_run(rc)
    arm = str(rc.get("arm", "fedavg"))
    cfg.arms = (arm,)
    return run_experiment(cfg, node_data(nodes), out, log=print)


def adapter_max_diff(a: Path, b: Path) -> float | None:
    fa, fb = a / "adapter_model.safetensors", b / "adapter_model.safetensors"
    if not (fa.exists() and fb.exists()):
        return None
    from safetensors.torch import load_file

    sa, sb = load_file(str(fa)), load_file(str(fb))
    if set(sa) != set(sb):
        return float("inf")
    return max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)


def diff(sim: Path, dist: Path, arm: str = "fedavg") -> str:
    rs = json.loads((sim / "results.json").read_text())["arms"][arm]
    rd = json.loads((dist / "results.json").read_text())["arms"][arm]

    def cards(path: Path) -> list[tuple[str, str, bool]]:
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        return [(r["set"], r["card"], all(r["matches"].values())) for r in rows if r["set"] != "injection_axis"]

    cs, cd = cards(sim / f"predictions_{arm}.jsonl"), cards(dist / f"predictions_{arm}.jsonl")
    assert [c[:2] for c in cs] == [c[:2] for c in cd], "the two runs scored different cards"
    agree = sum(a[2] == b[2] for a, b in zip(cs, cd, strict=True))
    mc = mcnemar_cards([c[2] for c in cd], [c[2] for c in cs])
    lines = [f"# Distributed vs simulated FedAvg ({arm})", "",
             "| Test set | simulated | distributed |", "|---|---|---|"]
    for name in rs["sets"]:
        s, d = rs["sets"][name], rd["sets"].get(name)
        lines.append(f"| {name} | {s['fields_correct']}/{s['fields_total']} ({s['field_accuracy']:.2f}) | "
                     + (f"{d['fields_correct']}/{d['fields_total']} ({d['field_accuracy']:.2f})" if d else "–") + " |")
    lines += ["", f"Cards with the same all-correct outcome: {agree}/{len(cs)}",
              f"Exact McNemar (distributed vs simulated, cards all correct): b={mc['b_first_only']} "
              f"c={mc['c_second_only']} p={mc['p_value']:.4g}"]
    md = adapter_max_diff(sim / "adapters" / arm, dist / "adapters" / arm)
    if md is not None:
        lines.append(f"Largest absolute difference between the two final adapters: {md:.3g}")
    t = rd.get("transport")
    if t:
        lines += ["", f"Distributed transport: {t['rounds_completed']} rounds, banks {', '.join(t['clients'])}, "
                      f"{t['bytes_per_upload'] / 1e6:.2f} MB per adapter message, {t['bytes_total'] / 1e6:.2f} MB in total, "
                      f"mean round {t['mean_round_seconds']:.1f} s"]
    text = "\n".join(lines) + "\n"
    (dist / f"compare_{arm}.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Distributed vs simulated FedAvg")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--nodes", type=Path, default=Path("deploy/nodes"))
    s.add_argument("--run-config", type=Path)
    s.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("diff")
    d.add_argument("--sim", type=Path, required=True)
    d.add_argument("--dist", type=Path, required=True)
    d.add_argument("--arm", default="fedavg")
    args = ap.parse_args()
    if args.cmd == "simulate":
        simulate(args.nodes, run_config(args.run_config), args.out)
    else:
        print(diff(args.sim, args.dist, args.arm))


if __name__ == "__main__":
    main()
