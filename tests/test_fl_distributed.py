"""Distributed FL (round 8): codec, aggregation rule, data isolation, and an end-to-end
run of a real SuperLink + three bank SuperNodes (separate processes, HTTP between
them) whose final adapter must equal the in-process simulation's when dropout = 0.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
flwr = pytest.importorskip("flwr")

from fl.data_gen import BANKS, build_datasets  # noqa: E402
from fl.distributed.codec import (  # noqa: E402
    NotAnAdapterError,
    payload_bytes,
    record_to_state,
    state_to_record,
)
from fl.distributed.prepare import load_eval_sets, load_examples, split_for_nodes  # noqa: E402
from fl.distributed.settings import config_from_run, round_seed  # noqa: E402
from fl.train import fedavg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def nodes(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from src.portal.samples import build

    if not (REPO / "data/portal_samples/axis_eligibility_2026-11.pdf").exists():
        build()
    out = tmp_path_factory.mktemp("nodes")
    data = build_datasets(train_cards_per_bank=6, test_cards_per_bank=2, variants_per_card=1, portal_dir=None)
    split_for_nodes(data, out)
    return out


def test_codec_round_trip_and_adapter_only_guard() -> None:
    state = {"base.layer.lora_A.weight": torch.randn(4, 8), "base.layer.lora_B.weight": torch.zeros(8, 4)}
    rec = state_to_record(state)
    back = record_to_state(rec)
    assert list(rec.keys()) == sorted(state) and all(torch.equal(back[k], state[k]) for k in state)
    assert payload_bytes(rec) == (32 + 32) * 4  # float32
    with pytest.raises(NotAnAdapterError):
        state_to_record({**state, "model.embed_tokens.weight": torch.zeros(2, 2)})  # base weights never leave


def test_flower_fedavg_equals_thesis_fedavg() -> None:
    from flwr.app import MetricRecord, RecordDict
    from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

    states = [{"x.lora_A.weight": torch.randn(3, 5)} for _ in range(3)]
    sizes = [180, 180, 90]
    replies = [RecordDict({"arrays": state_to_record(s), "metrics": MetricRecord({"num-examples": n})})
               for s, n in zip(states, sizes, strict=True)]
    flower = record_to_state(aggregate_arrayrecords(replies, "num-examples"))
    ours = fedavg(states, sizes)
    assert torch.allclose(flower["x.lora_A.weight"], ours["x.lora_A.weight"], atol=1e-6)


def test_each_node_folder_holds_only_its_own_data(nodes: Path) -> None:
    manifest = json.loads((nodes / "manifest.json").read_text())
    for bank in BANKS:
        rows = load_examples(nodes / bank / "train.jsonl")
        assert rows and {e.bank for e in rows} == {bank}
        assert manifest[f"{bank}/train.jsonl"]["banks"] == [bank]
        assert sorted(p.name for p in (nodes / bank).iterdir()) == ["train.jsonl"]
    server_files = [p.name for p in (nodes / "server").rglob("*") if p.is_file()]
    assert server_files and not any(n.startswith("train") for n in server_files)  # server: test sets only
    sets = load_eval_sets(nodes / "server" / "eval")
    assert list(sets)[0].startswith("synthetic_test/") and "injection_axis" in sets


def test_run_config_maps_to_flconfig_and_seeds_match_simulation() -> None:
    cfg = config_from_run({"model-name": "m", "num-server-rounds": 4, "local-steps": 3, "lora-r": 8, "quantize-4bit": False})
    assert (cfg.model_name, cfg.rounds, cfg.local_steps, cfg.lora_r, cfg.quantize_4bit) == ("m", 4, 3, 8, False)
    # fl.train.run_experiment uses seed * 1000 + rnd * 10 + client index, rnd counted from 0
    assert round_seed(cfg, server_round=1, bank_index=2) == cfg.seed * 1000 + 0 * 10 + 2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(port: int, timeout: float = 30) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.3)
    raise TimeoutError(f"port {port} never opened")


@pytest.mark.skipif(shutil.which("flower-superlink") is None, reason="Flower CLI not installed")
def test_distributed_run_reproduces_the_simulation(nodes: Path, tmp_path: Path) -> None:
    from fl.distributed.compare import adapter_max_diff, diff, run_config, simulate
    from fl.distributed.tiny_model import build

    model_dir = build(nodes, tmp_path / "tiny")
    overrides = {"model-name": str(model_dir), "quantize-4bit": False, "lora-r": 4, "lora-alpha": 8, "lora-dropout": 0.0,
                 "num-server-rounds": 2, "local-steps": 2, "batch-size": 1, "grad-accum": 1, "max-len": 1200,
                 "max-new-tokens": 4, "gen-batch": 4, "eval-limit": 1, "out-dir": str(tmp_path / "dist"),
                 "eval-dir": str(nodes / "server" / "eval")}
    rc_file = tmp_path / "run.toml"
    rc_file.write_text("".join(f"{k} = {json.dumps(v) if not isinstance(v, bool) else str(v).lower()}\n"
                               for k, v in overrides.items()))

    link_port = _free_port()
    env = {**os.environ, "PYTHONPATH": str(REPO), "FLWR_HOME": str(tmp_path / "flwr_home"),
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    (tmp_path / "flwr_home").mkdir()
    (tmp_path / "flwr_home" / "config.toml").write_text(
        f'[superlink]\ndefault = "t"\n[superlink.t]\naddress = "127.0.0.1:{link_port}"\ninsecure = true\n')
    bin_dir = Path(sys.executable).parent
    find = lambda name: shutil.which(name) or str(bin_dir / name)  # noqa: E731
    procs = [subprocess.Popen([find("flower-superlink"), "--insecure", "--port", str(link_port),
                               "--disable-runtime-dependency-installation"], env=env, cwd=REPO,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    try:
        _wait_port(link_port)
        for bank in BANKS:  # three bank nodes, each pointed at its own folder only
            procs.append(subprocess.Popen(
                [find("flower-supernode"), "--insecure", "--superlink", f"127.0.0.1:{link_port}",
                 "--port", str(_free_port()), "--node-config", f"bank='{bank}' data-dir='{nodes / bank}'"],
                env=env, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(6)
        run = subprocess.run([find("flwr"), "run", str(REPO / "fl/distributed/app"), "t", "--run-config", str(rc_file),
                              "--stream"], env=env, cwd=REPO, capture_output=True, text=True, timeout=600)
        assert "Received 3 results and 0 failures" in run.stdout + run.stderr, run.stdout[-2000:] + run.stderr[-2000:]
        for _ in range(60):  # the ServerApp writes results after the stream ends
            if (tmp_path / "dist" / "report.md").exists():
                break
            time.sleep(1)
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    dist = tmp_path / "dist"
    rounds = [json.loads(x) for x in (dist / "rounds_fedavg.jsonl").read_text().splitlines()]
    assert len(rounds) == 2 and all(sorted(r["banks"]) == sorted(BANKS) and not r["failures"] for r in rounds)
    assert all(b["bytes_up"] == rounds[0]["bytes_down_per_bank"] > 0 for r in rounds for b in r["banks"].values())
    results = json.loads((dist / "results.json").read_text())
    assert results["arms"]["fedavg"]["transport"]["rounds_completed"] == 2

    sim = tmp_path / "sim"
    simulate(nodes, run_config(rc_file), sim)
    assert adapter_max_diff(sim / "adapters" / "fedavg", dist / "adapters" / "fedavg") < 1e-5  # same FedAvg result
    assert "Cards with the same all-correct outcome" in diff(sim, dist)
