"""Split the RQ5 datasets into one folder per node.

    python -m fl.distributed.prepare --out deploy/nodes

writes

    deploy/nodes/SBI/train.jsonl        only SBI's training examples
    deploy/nodes/HDFC/train.jsonl       only HDFC's
    deploy/nodes/AXIS/train.jsonl       only Axis's
    deploy/nodes/server/eval/*.jsonl    public test sets only (no training data)
    deploy/nodes/manifest.json          file -> rows, banks, sha256

Each bank's container mounts ONLY its own folder, so isolation is enforced by
the deployment, not just by the code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from fl.data_gen import BANKS, Example, build_datasets
from fl.train import EVAL_SETS

TRAIN_FILE = "train.jsonl"


def load_examples(path: Path) -> list[Example]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            d["truth"] = {**d["truth"], "employment": tuple(d["truth"]["employment"])}
            rows.append(Example(**d))
    return rows


def _write(path: Path, rows: list[Example]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")
    return {"rows": len(rows), "banks": sorted({e.bank for e in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def split_for_nodes(data: dict[str, list[Example]], out: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for bank in BANKS:
        rows = data[f"train/{bank}"]
        assert all(e.bank == bank for e in rows), f"foreign example in {bank}'s training data"
        manifest[f"{bank}/{TRAIN_FILE}"] = _write(out / bank / TRAIN_FILE, rows)
    for name, rows in data.items():
        if name.split("/")[0] in EVAL_SETS:
            manifest[f"server/eval/{name.replace('/', '__')}.jsonl"] = _write(
                out / "server" / "eval" / f"{name.replace('/', '__')}.jsonl", rows)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_eval_sets(eval_dir: Path) -> dict[str, list[Example]]:
    """Server side: {'synthetic_test/SBI': [...], 'real_axis': [...], ...} in a fixed order."""
    out: dict[str, list[Example]] = {}
    files = sorted(eval_dir.glob("*.jsonl"),
                   key=lambda p: (EVAL_SETS.index(p.stem.split("__")[0]), p.stem))
    for path in files:
        out[path.stem.replace("__", "/")] = load_examples(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-node data folders for distributed FL")
    ap.add_argument("--out", type=Path, default=Path("deploy/nodes"))
    ap.add_argument("--no-portal-corrections", action="store_true")
    ap.add_argument("--small", action="store_true", help="tiny dataset for smoke tests")
    args = ap.parse_args()
    kwargs: dict[str, Any] = {"portal_dir": None if args.no_portal_corrections else Path("data/portal")}
    if args.small:
        kwargs.update(train_cards_per_bank=6, test_cards_per_bank=2, variants_per_card=1)
    manifest = split_for_nodes(build_datasets(**kwargs), args.out)
    for name, info in manifest.items():
        print(f"{name:45s} {info['rows']:4d} rows  banks={','.join(info['banks'])}")


if __name__ == "__main__":
    main()
