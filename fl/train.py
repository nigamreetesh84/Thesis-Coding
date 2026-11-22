"""Local LoRA fine-tuning, FedAvg aggregation and the RQ5 experiment arms.

Arms (all start from the same LoRA initialisation, same base model, same data):

  zero_shot      base model with the frozen extraction prompt, no training
  local/<BANK>   one bank trains alone on its own data       (rounds x local_steps steps)
  centralised    all banks' data pooled on one server        (rounds x local_steps x n_banks steps)
  fedavg         each bank trains locally, server averages adapters every round (FedAvg)
  fedavg_lobo    FedAvg with SBI and HDFC only; Axis is the unseen bank (leave-one-bank-out)

Compute is matched: every client does local_steps optimiser steps per round, and
the centralised arm does the same total number of steps as FedAvg over all clients.

Only LoRA adapter weights are exchanged in FedAvg (McMahan et al., 2017):
    w_global = sum_k (n_k / n) * w_k
No training example leaves its client.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fl.data_gen import BANKS, Example
from fl.evaluate import mcnemar_cards, score

EVAL_SETS = ("synthetic_test", "heldout_style", "real_thesis", "real_thesis_hard", "real_axis", "injection_axis")


@dataclass
class FLConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    quantize_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    lr: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 4
    rounds: int = 5
    local_steps: int = 15  # optimiser steps per client per round
    max_len: int = 1024
    max_new_tokens: int = 320
    gen_batch: int = 8
    seed: int = 7
    arms: tuple[str, ...] = ("zero_shot", "local", "centralised", "fedavg", "fedavg_lobo")
    lobo_bank: str = "AXIS"
    clients: tuple[str, ...] = BANKS
    eval_limit: int = 0  # >0: use only the first N cards of each eval set (smoke tests)


# ---------------------------------------------------------------------------
# Model, tokenizer, prompt format
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(cfg: FLConfig) -> tuple[Any, Any]:
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    kwargs: dict[str, Any] = {}
    if cfg.quantize_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    if "quantization_config" in kwargs:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # needed with gradient checkpointing; generation turns it back on
    return attach_lora(model, cfg), tokenizer


def attach_lora(model: Any, cfg: FLConfig) -> Any:
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(cfg.seed)
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.target_modules), task_type="CAUSAL_LM")
    return get_peft_model(model, lora)


def format_prompt(tokenizer: Any, prompt: str) -> str:
    """Same chat formatting as src/infrastructure/llm_client.py::LocalLLMClient._format_prompt."""
    system_marker = "SYSTEM:\n"
    messages = [{"role": "user", "content": prompt}]
    if prompt.startswith(system_marker):
        body_start = prompt.find("\n\n### ", len(system_marker))
        if body_start != -1:
            messages = [{"role": "system", "content": prompt[len(system_marker):body_start].strip()},
                        {"role": "user", "content": prompt[body_start:].strip()}]
    try:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    except Exception:  # template without a system role
        merged = [{"role": "user", "content": prompt}]
        return str(tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True))


def encode(tokenizer: Any, ex: Example, max_len: int) -> tuple[list[int], list[int]]:
    """input_ids and labels; loss only on the target JSON (+ end-of-sequence)."""
    p_ids = tokenizer(format_prompt(tokenizer, ex.prompt), add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(ex.target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(p_ids) + len(t_ids) > max_len:
        p_ids = p_ids[-(max_len - len(t_ids)):]  # keep the end of the prompt (cards list, task)
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids


# ---------------------------------------------------------------------------
# Adapter weights
# ---------------------------------------------------------------------------
def get_adapter(model: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    return {k: v.detach().to("cpu").clone() for k, v in get_peft_model_state_dict(model).items()}


def set_adapter(model: Any, state: dict[str, Any]) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model, state)


def fedavg(states: list[dict[str, Any]], weights: list[float]) -> dict[str, Any]:
    """Weighted average of adapter state dicts (weights = client example counts)."""
    total = float(sum(weights))
    out: dict[str, Any] = {}
    for key in states[0]:
        acc = None
        for state, w in zip(states, weights, strict=True):
            term = state[key].float() * (w / total)
            acc = term if acc is None else acc + term
        out[key] = acc.to(states[0][key].dtype)  # type: ignore[union-attr]
    return out


# ---------------------------------------------------------------------------
# Training and generation
# ---------------------------------------------------------------------------
def train_steps(model: Any, tokenizer: Any, examples: list[Example], steps: int, cfg: FLConfig,
                rng: random.Random, log: Callable[[str], None] = print) -> list[float]:
    """`steps` optimiser steps of AdamW on LoRA params; returns the loss per step."""
    import torch

    if steps <= 0 or not examples:
        return []
    device = next(p for p in model.parameters() if p.requires_grad).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    encoded = [encode(tokenizer, ex, cfg.max_len) for ex in examples]
    order: list[int] = []
    losses: list[float] = []
    model.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            batch = []
            for _ in range(cfg.batch_size):
                if not order:
                    order = list(range(len(encoded)))
                    rng.shuffle(order)
                batch.append(encoded[order.pop()])
            width = max(len(ids) for ids, _ in batch)
            input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in batch], device=device)
            labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in batch], device=device)
            attn = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch], device=device)
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss / cfg.grad_accum
            if not torch.isfinite(loss):
                log(f"  non-finite loss at step {step}; batch skipped")
                continue
            loss.backward()
            step_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(step_loss)
    return losses


def generate(model: Any, tokenizer: Any, prompts: list[str], cfg: FLConfig) -> list[str]:
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    model.eval()
    outputs: list[str] = []
    with torch.no_grad():
        for i in range(0, len(prompts), cfg.gen_batch):
            texts = [format_prompt(tokenizer, p) for p in prompts[i:i + cfg.gen_batch]]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id, use_cache=True)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                outputs.append(tokenizer.decode(row, skip_special_tokens=True))
    model.train()
    return outputs


def evaluate_arm(model: Any, tokenizer: Any, data: dict[str, list[Example]], cfg: FLConfig) -> dict[str, Any]:
    sets: dict[str, list[Example]] = {}
    for name, rows in data.items():
        family = name.split("/")[0]
        if family in EVAL_SETS:
            sets[name] = rows[: cfg.eval_limit] if cfg.eval_limit else rows
    prompts = [ex.prompt for rows in sets.values() for ex in rows]
    raws = generate(model, tokenizer, prompts, cfg) if prompts else []
    result: dict[str, Any] = {"sets": {}, "rows": {}}
    i = 0
    for name, rows in sets.items():
        scored = [{"bank": ex.bank, "card": ex.card, "raw": raws[i + j], "truth": ex.truth} for j, ex in enumerate(rows)]
        i += len(rows)
        result["sets"][name] = score(scored)
        result["rows"][name] = [{**r, "truth": {**r["truth"], "employment": list(r["truth"]["employment"])}} for r in scored]
    return result


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_experiment(cfg: FLConfig, data: dict[str, list[Example]], out_dir: Path, model: Any = None,
                   tokenizer: Any = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run every arm in cfg.arms; results.json is rewritten after each arm (resumable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results: dict[str, Any] = json.loads(results_path.read_text()) if results_path.exists() else {}
    results["config"] = {**asdict(cfg), "target_modules": list(cfg.target_modules), "arms": list(cfg.arms),
                         "clients": list(cfg.clients)}
    results.setdefault("arms", {})
    if model is None:
        model, tokenizer = load_model_and_tokenizer(cfg)
    init_state = get_adapter(model)
    client_data = {b: data[f"train/{b}"] for b in cfg.clients}
    total_local = cfg.rounds * cfg.local_steps

    def save() -> None:
        results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    def finish(arm: str, losses: Any, started: float, extra: dict[str, Any] | None = None) -> None:
        log(f"[{arm}] evaluating")
        ev = evaluate_arm(model, tokenizer, data, cfg)
        with (out_dir / f"predictions_{arm.replace('/', '_')}.jsonl").open("w", encoding="utf-8") as f:
            for set_name, rows in ev["rows"].items():
                for r in rows:
                    f.write(json.dumps({"set": set_name, **r}, default=str) + "\n")
        results["arms"][arm] = {"sets": ev["sets"], "losses": losses, "seconds": round(time.time() - started, 1),
                                **(extra or {})}
        try:
            model.save_pretrained(str(out_dir / "adapters" / arm.replace("/", "_")))
        except Exception as exc:  # saving is a convenience; never lose the results over it
            log(f"  adapter not saved: {exc}")
        save()
        log(f"[{arm}] done: " + ", ".join(f"{k}={v['field_accuracy']:.3f}" for k, v in ev["sets"].items()))

    def pending(arm: str) -> bool:
        if arm in results["arms"]:
            log(f"[{arm}] already in results.json, skipped")
            return False
        return True

    rng = random.Random(cfg.seed)
    for arm in cfg.arms:
        if arm == "zero_shot" and pending("zero_shot"):
            started = time.time()
            set_adapter(model, init_state)  # LoRA B = 0 at init: identical to the base model
            finish("zero_shot", [], started)
        elif arm == "local":
            for bank in cfg.clients:
                name = f"local/{bank}"
                if not pending(name):
                    continue
                started = time.time()
                set_adapter(model, init_state)
                log(f"[{name}] {total_local} steps on {len(client_data[bank])} examples")
                losses = train_steps(model, tokenizer, client_data[bank], total_local, cfg, random.Random(cfg.seed), log)
                finish(name, losses, started, {"train_examples": len(client_data[bank])})
        elif arm == "centralised" and pending("centralised"):
            started = time.time()
            set_adapter(model, init_state)
            pooled = [ex for b in cfg.clients for ex in client_data[b]]
            steps = total_local * len(cfg.clients)
            log(f"[centralised] {steps} steps on {len(pooled)} pooled examples")
            losses = train_steps(model, tokenizer, pooled, steps, cfg, random.Random(cfg.seed), log)
            finish("centralised", losses, started, {"train_examples": len(pooled)})
        elif arm in ("fedavg", "fedavg_lobo") and pending(arm):
            started = time.time()
            clients = [b for b in cfg.clients if not (arm == "fedavg_lobo" and b == cfg.lobo_bank)]
            global_state = copy.deepcopy(init_state)
            history: list[dict[str, Any]] = []
            for rnd in range(cfg.rounds):
                states, weights, round_losses = [], [], {}
                for bank in clients:
                    set_adapter(model, global_state)  # client starts from the current global adapter
                    losses = train_steps(model, tokenizer, client_data[bank], cfg.local_steps, cfg,
                                         random.Random(cfg.seed * 1000 + rnd * 10 + clients.index(bank)), log)
                    states.append(get_adapter(model))
                    weights.append(len(client_data[bank]))
                    round_losses[bank] = losses
                global_state = fedavg(states, weights)  # only adapter weights reach the server
                mean_last = {b: (sum(v[-3:]) / len(v[-3:]) if v else math.nan) for b, v in round_losses.items()}
                history.append({"round": rnd + 1, "client_losses": round_losses, "mean_last_loss": mean_last})
                log(f"[{arm}] round {rnd + 1}/{cfg.rounds} " + " ".join(f"{b}:{v:.3f}" for b, v in mean_last.items()))
            set_adapter(model, global_state)
            finish(arm, history, started, {"clients": clients,
                                           "bytes_per_upload": sum(v.numel() * v.element_size() for v in global_state.values())})
    results["comparisons"] = compare(results)
    save()
    return results


def compare(results: dict[str, Any]) -> dict[str, Any]:
    """Headline comparisons on pooled per-card outcomes (exact McNemar)."""
    arms = results.get("arms", {})

    def pooled(arm: str, families: tuple[str, ...]) -> tuple[list[bool], int, int]:
        outcomes: list[bool] = []
        fields_ok = fields_total = 0
        for name, s in sorted(arms[arm]["sets"].items()):
            if name.split("/")[0] in families:
                outcomes += [True] * s["card_all_correct"] + [False] * (s["n_cards"] - s["card_all_correct"])
                fields_ok += s["fields_correct"]
                fields_total += s["fields_total"]
        return outcomes, fields_ok, fields_total

    out: dict[str, Any] = {"summary": {}}
    families = {"all_unseen": EVAL_SETS[:-1], "real_documents": ("real_thesis", "real_thesis_hard", "real_axis"),
                "synthetic": ("synthetic_test", "heldout_style")}
    for arm in arms:
        out["summary"][arm] = {}
        for fam, sets in families.items():
            _, ok, tot = pooled(arm, sets)
            out["summary"][arm][fam] = {"fields_correct": ok, "fields_total": tot,
                                        "field_accuracy": ok / tot if tot else 0.0}
    # paired tests need identical card order: sets are scored in a fixed order, so the pooled lists align
    pairs = [("fedavg", "zero_shot"), ("fedavg", "centralised"), ("fedavg", "fedavg_lobo")]
    pairs += [("fedavg", a) for a in arms if a.startswith("local/")]
    out["pairs"] = [list(p) for p in pairs if p[0] in arms and p[1] in arms]
    return out


def per_card_mcnemar(out_dir: Path, pairs: list[list[str]], families: tuple[str, ...] = EVAL_SETS[:-1]) -> dict[str, Any]:
    """Exact McNemar on paired per-card all-correct outcomes, read from the predictions files."""
    def outcomes(arm: str) -> list[tuple[str, str, bool]]:
        rows = []
        path = out_dir / f"predictions_{arm.replace('/', '_')}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["set"].split("/")[0] in families:
                rows.append((r["set"], r["card"], all(r["matches"].values())))
        return rows

    out = {}
    for a, b in pairs:
        ra, rb = outcomes(a), outcomes(b)
        assert [(s, c) for s, c, _ in ra] == [(s, c) for s, c, _ in rb], "card order differs between arms"
        out[f"{a} vs {b}"] = mcnemar_cards([x for *_, x in ra], [x for *_, x in rb])
    return out


__all__ = ["FLConfig", "fedavg", "run_experiment", "per_card_mcnemar"]
