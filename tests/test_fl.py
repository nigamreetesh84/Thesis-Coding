"""Federated-learning pipeline (RQ5): data, scoring, FedAvg maths and an end-to-end CPU run.

The end-to-end test uses a tiny randomly initialised model and a tokenizer trained
on the fly, so it needs no download and no GPU. It checks the MECHANICS (every arm
runs, only adapter weights are averaged, results are written and resumable); the
real numbers come from the Colab run with Qwen2.5-1.5B.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from fl.data_gen import BANKS, build_datasets  # noqa: E402
from fl.evaluate import field_matches, mcnemar_cards, score  # noqa: E402
from fl.train import FLConfig, attach_lora, fedavg, get_adapter, per_card_mcnemar, run_experiment, set_adapter  # noqa: E402


@pytest.fixture(scope="module")
def data() -> dict:
    from src.portal.samples import build

    if not Path("data/portal_samples/axis_eligibility_2026-11.pdf").exists():
        build()
    return build_datasets(train_cards_per_bank=6, test_cards_per_bank=2, variants_per_card=1, portal_dir=None)


def test_datasets_are_per_bank_and_disjoint(data: dict) -> None:
    for bank in BANKS:
        assert data[f"train/{bank}"] and all(e.bank == bank for e in data[f"train/{bank}"])
        train_cards = {e.card for e in data[f"train/{bank}"]}
        assert not train_cards & {e.card for e in data[f"synthetic_test/{bank}"]}  # unseen cards
        assert all(e.style == bank for e in data[f"train/{bank}"])  # each bank's own house style
    assert len(data["real_thesis"]) == 8 and len(data["real_thesis_hard"]) == 8 and len(data["real_axis"]) == 4
    assert all("### TASK: EXTRACT REQUIREMENTS" in e.prompt for e in data["real_axis"])  # frozen prompt
    target = json.loads(data["train/HDFC"][0].target)[0]
    assert set(target["requirements"]) == {"min_annual_income_inr", "min_age", "min_cibil", "allowed_employment",
                                           "existing_relationship_required"}


def test_gold_targets_score_perfectly(data: dict) -> None:
    rows = [{"bank": e.bank, "card": e.card, "raw": e.target, "truth": e.truth}
            for name in ("train/SBI", "synthetic_test/AXIS", "heldout_style/HDFC") for e in data[name]]
    s = score(rows)
    assert s["field_accuracy"] == 1.0 and s["json_valid_rate"] == 1.0


def test_unstated_threshold_must_stay_null() -> None:
    truth = {"min_income": None, "min_age": 18, "min_cibil": None, "employment": ("any",), "relationship_required": False}
    invented = {"min_income": 600000, "min_age": 18, "min_cibil": 750, "allowed_employment": "any", "relationship_required": False}
    m = field_matches(invented, truth)
    assert not m["min_income"] and not m["min_cibil"] and m["min_age"]
    assert mcnemar_cards([True, True, False], [False, True, False])["b_first_only"] == 1


def test_fedavg_is_the_weighted_mean() -> None:
    a = {"w": torch.tensor([0.0, 2.0])}
    b = {"w": torch.tensor([4.0, 6.0])}
    avg = fedavg([a, b], [1, 3])
    assert torch.allclose(avg["w"], torch.tensor([3.0, 5.0]))


def _tiny_model_and_tokenizer(data: dict):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    texts = [e.prompt + e.target for rows in data.values() for e in rows][:200]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=600, special_tokens=["<unk>", "<pad>", "<eos>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
                               "{% if add_generation_prompt %}<assistant>{% endif %}")
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=2048,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config), tokenizer


def test_end_to_end_tiny_run_is_complete_and_resumable(data: dict, tmp_path: Path) -> None:
    cfg = FLConfig(model_name="tiny", quantize_4bit=False, rounds=2, local_steps=2, batch_size=1, grad_accum=1,
                   max_new_tokens=8, gen_batch=4, eval_limit=1, lora_r=4, lora_alpha=8, max_len=1200)
    base, tokenizer = _tiny_model_and_tokenizer(data)
    model = attach_lora(base, cfg)
    init = get_adapter(model)
    assert all("lora" in k for k in init)  # only adapter weights would ever leave a client

    results = run_experiment(cfg, data, tmp_path, model=model, tokenizer=tokenizer, log=lambda _: None)
    assert set(results["arms"]) == {"zero_shot", "local/SBI", "local/HDFC", "local/AXIS", "centralised", "fedavg", "fedavg_lobo"}
    assert results["arms"]["fedavg_lobo"]["clients"] == ["SBI", "HDFC"]
    assert len(results["arms"]["fedavg"]["losses"]) == cfg.rounds
    assert len(results["arms"]["centralised"]["losses"]) == cfg.rounds * cfg.local_steps * 3  # compute matched
    for arm in results["arms"].values():
        assert "real_axis" in arm["sets"] and "heldout_style/AXIS" in arm["sets"]

    # adapters changed with training, and are restored exactly for the next arm
    set_adapter(model, init)
    assert all(torch.equal(get_adapter(model)[k], init[k]) for k in init)

    stats = per_card_mcnemar(tmp_path, [["fedavg", "zero_shot"], ["fedavg", "local/AXIS"]])
    assert set(stats) == {"fedavg vs zero_shot", "fedavg vs local/AXIS"}

    # resumable: a second call skips finished arms
    seen: list[str] = []
    run_experiment(cfg, data, tmp_path, model=model, tokenizer=tokenizer, log=seen.append)
    assert any("already in results.json" in s for s in seen)

    from fl.run_fl import report

    text = report(tmp_path)
    assert "| real_axis |" in text and "Exact McNemar" in text


def test_training_reduces_loss_on_a_fixed_batch(data: dict) -> None:
    from fl.train import train_steps

    cfg = FLConfig(quantize_4bit=False, batch_size=1, grad_accum=1, lr=5e-3, lora_r=8, lora_alpha=16, max_len=1200)
    base, tokenizer = _tiny_model_and_tokenizer(data)
    model = attach_lora(base, cfg)
    losses = train_steps(model, tokenizer, data["train/SBI"][:1], 25, cfg, random.Random(0))
    assert losses[-1] < losses[0]
