"""Flower run config (pyproject.toml [tool.flwr.app.config]) <-> FLConfig.

The keys mirror FLConfig so a distributed run can use exactly the settings of a
simulated run (Run 1 or the ``quick`` preset).
"""

from __future__ import annotations

from typing import Any

from fl.train import FLConfig

# run-config key -> (FLConfig field, type)
KEYS: dict[str, tuple[str, type]] = {
    "model-name": ("model_name", str),
    "quantize-4bit": ("quantize_4bit", bool),
    "lora-r": ("lora_r", int),
    "lora-alpha": ("lora_alpha", int),
    "lora-dropout": ("lora_dropout", float),
    "lr": ("lr", float),
    "batch-size": ("batch_size", int),
    "grad-accum": ("grad_accum", int),
    "num-server-rounds": ("rounds", int),
    "local-steps": ("local_steps", int),
    "max-len": ("max_len", int),
    "max-new-tokens": ("max_new_tokens", int),
    "gen-batch": ("gen_batch", int),
    "seed": ("seed", int),
    "eval-limit": ("eval_limit", int),
}


def config_from_run(run_config: dict[str, Any]) -> FLConfig:
    cfg = FLConfig()
    for key, (field, typ) in KEYS.items():
        if key in run_config:
            setattr(cfg, field, typ(run_config[key]))
    return cfg


def round_seed(cfg: FLConfig, server_round: int, bank_index: int) -> int:
    """Same per-client, per-round seed as the simulation (fl.train.run_experiment, FedAvg arms)."""
    return cfg.seed * 1000 + (server_round - 1) * 10 + bank_index


__all__ = ["KEYS", "config_from_run", "round_seed"]
