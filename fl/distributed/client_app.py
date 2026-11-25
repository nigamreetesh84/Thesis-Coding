"""Flower ClientApp: the bank side of distributed FedAvg.

Started inside a bank's SuperNode, e.g.

    flower-supernode --superlink <host>:9092 \
        --node-config "bank='SBI' data-dir='/data'"

For each ``train`` message it: loads the global adapter it was sent, runs
``local-steps`` optimiser steps on <data-dir>/train.jsonl (this bank's file and
nothing else), and replies with the updated adapter and scalar metrics.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl.data_gen import BANKS
from fl.distributed.codec import payload_bytes, record_to_state, state_to_record
from fl.distributed.prepare import TRAIN_FILE, load_examples
from fl.distributed.settings import config_from_run, round_seed
from fl.train import get_adapter, load_model_and_tokenizer, set_adapter, train_steps

app = ClientApp()

_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}  # the base model is loaded once per process


def _model(cfg: Any) -> tuple[Any, Any, float]:
    key = (cfg.model_name, cfg.quantize_4bit, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout, tuple(cfg.target_modules))
    started = time.time()
    if key not in _CACHE:
        _CACHE[key] = load_model_and_tokenizer(cfg)
    model, tokenizer = _CACHE[key]
    return model, tokenizer, time.time() - started


def bank_identity(context: Context) -> tuple[str, Path]:
    bank = str(context.node_config["bank"]).upper()
    if bank not in BANKS:
        raise ValueError(f"unknown bank {bank!r}; expected one of {BANKS}")
    return bank, Path(str(context.node_config["data-dir"]))


@app.train()
def train(msg: Message, context: Context) -> Message:
    bank, data_dir = bank_identity(context)
    cfg = config_from_run(dict(context.run_config))
    server_round = int(msg.content["config"]["server-round"])
    examples = load_examples(data_dir / TRAIN_FILE)  # this bank's own file only
    if any(e.bank != bank for e in examples):
        raise ValueError(f"{data_dir / TRAIN_FILE} holds examples of another bank")

    model, tokenizer, load_seconds = _model(cfg)
    incoming: ArrayRecord = msg.content["arrays"]
    set_adapter(model, record_to_state(incoming))

    import torch

    seed = round_seed(cfg, server_round, BANKS.index(bank))
    torch.manual_seed(seed)
    started = time.time()
    losses = train_steps(model, tokenizer, examples, cfg.local_steps, cfg, random.Random(seed), log=lambda _: None)
    train_seconds = time.time() - started

    outgoing = state_to_record(get_adapter(model))
    tail = losses[-3:] or [float("nan")]
    metrics = MetricRecord({
        "num-examples": len(examples),  # FedAvg weight n_k
        "train-loss": float(sum(tail) / len(tail)),
        "train-seconds": float(train_seconds),
        "load-seconds": float(load_seconds),
        "bytes-down": payload_bytes(incoming),
        "bytes-up": payload_bytes(outgoing),
        "bank-index": BANKS.index(bank),
    })
    info = ConfigRecord({"bank": bank, "losses": [float(x) for x in losses]})
    return Message(content=RecordDict({"arrays": outgoing, "metrics": metrics, "info": info}), reply_to=msg)
