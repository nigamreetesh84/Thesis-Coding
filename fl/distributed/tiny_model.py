"""A tiny, randomly initialised Qwen2 model + tokenizer saved to a local folder.

Used to smoke-test the distributed system end to end without downloading a
model (accuracy is meaningless; the point is that processes, messages, FedAvg,
logging and evaluation all work). Same construction as tests/test_fl.py.

    python -m fl.distributed.tiny_model --data deploy/nodes --out deploy/models/tiny-qwen2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fl.distributed.prepare import load_examples


def build(data_dir: Path, out: Path, vocab_size: int = 600, seed: int = 0) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    texts = [e.prompt + e.target for p in sorted(data_dir.rglob("*.jsonl")) for e in load_examples(p)][:200]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<eos>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
                               "{% if add_generation_prompt %}<assistant>{% endif %}")
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=2048,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    torch.manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    Qwen2ForCausalLM(config).save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Save a tiny offline model for distributed smoke tests")
    ap.add_argument("--data", type=Path, default=Path("deploy/nodes"))
    ap.add_argument("--out", type=Path, default=Path("deploy/models/tiny-qwen2"))
    args = ap.parse_args()
    print(build(args.data, args.out))


if __name__ == "__main__":
    main()
