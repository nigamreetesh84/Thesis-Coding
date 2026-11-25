"""Flower ServerApp: aggregation server of distributed FedAvg.

Runs inside the SuperLink when the researcher calls ``flwr run``. It

1. builds the initial LoRA adapter (same seed as the simulation, B = 0, so
   round 0 equals the base model),
2. runs FedAvg for ``num-server-rounds`` rounds over the connected bank nodes
   (weights = each bank's ``num-examples``, the same rule as fl.train.fedavg),
3. logs every round to <out-dir>/rounds.jsonl (bytes down/up, seconds, loss per bank),
4. saves the final adapter and, if ``server-eval`` is true, scores it on the
   public test sets in <eval-dir> and writes results.json / report.md in the
   same format as Run 1 (so ``python -m fl.run_fl report`` works on it).

The server holds no training data: <eval-dir> contains only the test sets.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl.distributed.codec import payload_bytes, record_to_state, state_to_record
from fl.distributed.settings import config_from_run

app = ServerApp()


def _flwr_version() -> str:
    import flwr

    return str(flwr.__version__)


class LoggedFedAvg(FedAvg):
    """Flower FedAvg + one JSON line per round with what actually crossed the network."""

    def __init__(self, round_log: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.round_log = round_log
        self.rounds: list[dict[str, Any]] = []
        self._sent_at = 0.0
        self._down = 0

    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord,
                        grid: Grid) -> Iterable[Message]:
        self._sent_at = time.time()
        self._down = payload_bytes(arrays)
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round: int,
                        replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        received = time.time()
        banks: dict[str, Any] = {}
        failures: list[dict[str, Any]] = []
        for msg in replies:
            if msg.has_error():
                failures.append({"node_id": msg.metadata.src_node_id, "reason": msg.error.reason})
                continue
            m = msg.content["metrics"]
            bank = str(msg.content["info"]["bank"]) if "info" in msg.content else str(msg.metadata.src_node_id)
            banks[bank] = {"node_id": msg.metadata.src_node_id, "num_examples": int(m["num-examples"]),
                           "train_loss": float(m["train-loss"]), "train_seconds": round(float(m["train-seconds"]), 2),
                           "load_seconds": round(float(m["load-seconds"]), 2), "bytes_down": int(m["bytes-down"]),
                           "bytes_up": int(m["bytes-up"]),
                           "losses": list(msg.content["info"]["losses"]) if "info" in msg.content else []}
        started = time.time()
        arrays, metrics = super().aggregate_train(server_round, replies)
        entry = {"round": server_round, "banks": dict(sorted(banks.items())), "failures": failures,
                 "bytes_down_per_bank": self._down,
                 "bytes_up_total": sum(b["bytes_up"] for b in banks.values()),
                 "aggregation_seconds": round(time.time() - started, 3),
                 "round_seconds": round(received - self._sent_at, 2)}
        self.rounds.append(entry)
        self.round_log.parent.mkdir(parents=True, exist_ok=True)
        with self.round_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return arrays, metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = dict(context.run_config)
    cfg = config_from_run(rc)
    arm = str(rc.get("arm", "fedavg"))
    out_dir = Path(str(rc.get("out-dir", "results_fl/distributed")))
    eval_dir = Path(str(rc.get("eval-dir", "deploy/nodes/server/eval")))
    min_nodes = int(rc.get("min-nodes", 3))
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    from fl.train import get_adapter, load_model_and_tokenizer, set_adapter

    # model-name = a Hugging Face id, or a local folder (e.g. the offline tiny model from fl.distributed.tiny_model)
    model, tokenizer = load_model_and_tokenizer(cfg)
    initial = state_to_record(get_adapter(model))

    round_log = out_dir / f"rounds_{arm}.jsonl"
    round_log.unlink(missing_ok=True)
    strategy = LoggedFedAvg(round_log=round_log, fraction_train=1.0, fraction_evaluate=0.0,
                            min_train_nodes=min_nodes, min_available_nodes=min_nodes)
    result = strategy.start(grid=grid, initial_arrays=initial, num_rounds=cfg.rounds,
                            timeout=float(rc.get("round-timeout", 3600)),
                            train_config=ConfigRecord({"lr": cfg.lr, "local-steps": cfg.local_steps}))
    set_adapter(model, record_to_state(result.arrays))
    train_seconds = time.time() - started
    model.save_pretrained(str(out_dir / "adapters" / arm))

    clients = sorted({b for r in strategy.rounds for b in r["banks"]})
    transport = {
        "framework": "flower " + _flwr_version(), "transport": "Flower Fleet API over HTTP (HTTPS when TLS is on)", "rounds_completed": len(strategy.rounds),
        "clients": clients, "bytes_per_upload": strategy.rounds[0]["bytes_down_per_bank"] if strategy.rounds else 0,
        "bytes_total": sum(r["bytes_down_per_bank"] * len(r["banks"]) + r["bytes_up_total"] for r in strategy.rounds),
        "mean_round_seconds": (sum(r["round_seconds"] for r in strategy.rounds) / len(strategy.rounds)
                               if strategy.rounds else 0.0),
        "train_seconds": round(train_seconds, 1),
    }
    (out_dir / f"transport_{arm}.json").write_text(json.dumps(transport, indent=2), encoding="utf-8")

    if bool(rc.get("server-eval", True)):
        evaluate_and_record(model, tokenizer, cfg, arm, eval_dir, out_dir, strategy.rounds, transport, started)


def evaluate_and_record(model: Any, tokenizer: Any, cfg: Any, arm: str, eval_dir: Path, out_dir: Path,
                        rounds: list[dict[str, Any]], transport: dict[str, Any], started: float) -> dict[str, Any]:
    """Score the global adapter on the server's public test sets; write Run-1-format results."""
    from dataclasses import asdict

    from fl.distributed.prepare import load_eval_sets
    from fl.run_fl import report
    from fl.train import compare, evaluate_arm

    data = load_eval_sets(eval_dir)
    ev = evaluate_arm(model, tokenizer, data, cfg)
    with (out_dir / f"predictions_{arm}.jsonl").open("w", encoding="utf-8") as f:
        for set_name, rows in ev["rows"].items():
            for r in rows:
                f.write(json.dumps({"set": set_name, **r}, default=str) + "\n")
    path = out_dir / "results.json"
    results: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    results["config"] = {**asdict(cfg), "target_modules": list(cfg.target_modules), "deployment": "distributed"}
    results.setdefault("arms", {})[arm] = {
        "sets": ev["sets"], "losses": [{"round": r["round"], "client_losses": {b: v["losses"] for b, v in r["banks"].items()}}
                                       for r in rounds],
        "seconds": round(time.time() - started, 1), "clients": transport["clients"],
        "bytes_per_upload": transport["bytes_per_upload"], "transport": transport}
    results["comparisons"] = compare(results)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    report(out_dir)
    return results
