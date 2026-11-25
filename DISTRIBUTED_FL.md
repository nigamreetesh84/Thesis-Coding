# Distributed federated learning (RQ5, round 8)

Run 1 simulated FedAvg in one process (`fl/train.py`). This round runs the **same
training code and the same FedAvg rule** as a real distributed system on the
[Flower](https://flower.ai) Deployment Engine (flwr 1.45):

| Part | Process | Holds | Sends |
|---|---|---|---|
| SuperLink + ServerApp (`fl/distributed/server_app.py`) | aggregation server | global LoRA adapter; public test sets only | global adapter to each bank, every round |
| SuperNode + ClientApp (`fl/distributed/client_app.py`), one per bank | SBI, HDFC, AXIS | base model + that bank's `train.jsonl` only | updated adapter + `num-examples` |
| `flwr run` | researcher's CLI | the run config | starts a run on the SuperLink |

Transport: Flower Fleet API over HTTP on port 8000 (HTTPS with `TLS=1`). Only
LoRA tensors can be sent (`fl/distributed/codec.py` refuses anything else).

## Files

| File | Role |
|---|---|
| `fl/distributed/app/pyproject.toml` | Flower app; defaults = Run 1 settings |
| `fl/distributed/app/quick.toml`, `tiny.toml` | D1 settings (Qwen2.5-0.5B) and an offline smoke test |
| `fl/distributed/prepare.py` | `deploy/nodes/<BANK>/train.jsonl` + `deploy/nodes/server/eval/*.jsonl` + manifest |
| `fl/distributed/compare.py` | simulated run on the same folders + side-by-side diff |
| `fl/distributed/tiny_model.py` | tiny random model for offline tests |
| `scripts/run_distributed_local.sh` | setup A: separate processes, one machine |
| `scripts/run_distributed_netns.sh` | setup A+: each node in its own Linux network namespace (own IP) and Unix user |
| `scripts/run_distributed_docker.sh`, `deploy/docker-compose.yml`, `deploy/Dockerfile` | setup B: one container per node |
| `deploy/certs/make_certs.sh` | CA + server certificate for TLS |
| `tests/test_fl_distributed.py` | codec, aggregation rule, isolation, end-to-end run vs simulation |

## Outputs (in `out-dir`)

`rounds_<arm>.jsonl` (per round: bytes down/up and seconds per bank, failures),
`transport_<arm>.json`, `results.json` + `predictions_<arm>.jsonl` + `report.md`
(same format as Run 1), `adapters/<arm>/`, and `compare_<arm>.md` after a diff.

## Commands

```bash
pip install "flwr>=1.45,<2"                      # plus requirements.txt
python -m fl.distributed.prepare --out deploy/nodes

# smoke test, offline (tiny random model): 3 bank processes + server
bash scripts/run_distributed_local.sh

# D1: Qwen2.5-0.5B, 3 rounds x 8 steps (CPU is fine, about 30-60 min)
RUN_CONFIG=fl/distributed/app/quick.toml bash scripts/run_distributed_local.sh
RUN_CONFIG=fl/distributed/app/quick.toml BANKS="SBI HDFC" MIN_NODES=2 ARM=fedavg_lobo \
  bash scripts/run_distributed_local.sh
# same settings, simulated, then the comparison table for the thesis
python -m fl.distributed.compare simulate --run-config fl/distributed/app/quick.toml --out results_fl/sim_quick
python -m fl.distributed.compare diff --sim results_fl/sim_quick --dist results_fl/distributed_quick

# with TLS
TLS=1 RUN_CONFIG=fl/distributed/app/quick.toml bash scripts/run_distributed_local.sh

# Linux, root: every node in its own network namespace + Unix user (writes isolation evidence)
sudo TLS=1 RUN_CONFIG=fl/distributed/app/quick.toml bash scripts/run_distributed_netns.sh

# Docker (Windows/Mac: Docker Desktop; run from Git Bash or WSL)
RUN_CONFIG=fl/distributed/app/quick.toml bash scripts/run_distributed_docker.sh
```

### Several machines (setup C, e.g. full Run 1 settings on GPUs)

On the server host (must be reachable from the banks on port 8000):

```bash
bash deploy/certs/make_certs.sh <server-public-ip-or-dns>
flower-superlink --ssl-ca-certfile deploy/certs/ca.crt --ssl-certfile deploy/certs/server.crt \
  --ssl-keyfile deploy/certs/server.key --host 0.0.0.0 --disable-runtime-dependency-installation
```

On each bank machine (repo checked out, `python -m fl.distributed.prepare` run, then keep only that bank's folder):

```bash
export PYTHONPATH=$(pwd)
flower-supernode --root-certificates ca.crt --superlink <server>:8000 \
  --node-config "bank='SBI' data-dir='deploy/nodes/SBI'"
```

On the researcher's machine: add a connection to `$FLWR_HOME/config.toml`
(`address = "<server>:8000"`, `root-certificates = "/abs/path/ca.crt"`), then
`PYTHONPATH=. flwr run fl/distributed/app <connection> --stream`.

## Verified here (25 Nov 2026, CPU, tiny offline model)

* Setup A with and without TLS, and the leave-one-bank-out run (2 banks): all rounds 3/3 (or 2/2) results, 0 failures.
* Setup A+ (network namespaces 10.42.0.10-13, separate users, TLS): run completed; a bank's
  process gets "Permission denied" on another bank's file, and so does the server.
* With `lora-dropout = 0` the distributed final adapter equals the simulated one to 3e-8
  (largest absolute difference), i.e. the deployment reproduces FedAvg exactly.
* Docker: compose file validated (`docker compose config`); the image was not built here
  (no Docker Hub access in the build sandbox).
* Qwen2.5 runs (D1/D2) need a Hugging Face download, not available in the sandbox: run them locally/Colab.

## Limitations

Research prototype: few nodes, one operator; no differential privacy or secure
aggregation (Flower provides DP wrappers, not used); the ClientApp is started per
message, so each round reloads the base model (`load_seconds` in the round log).
