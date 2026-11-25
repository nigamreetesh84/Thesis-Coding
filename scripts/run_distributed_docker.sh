#!/usr/bin/env bash
# Setup B: SuperLink + bank SuperNodes as Docker containers on a private network (flnet).
#
#   bash scripts/run_distributed_docker.sh                                   # tiny offline model
#   RUN_CONFIG=fl/distributed/app/quick.toml bash scripts/run_distributed_docker.sh   # D1 (Qwen2.5-0.5B, CPU)
#   LOBO=1 RUN_CONFIG=fl/distributed/app/quick.toml bash scripts/run_distributed_docker.sh  # Axis left out
#
# Needs Docker with Compose v2, and `pip install "flwr>=1.45,<2"` on the host for the CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
RUN_CONFIG=${RUN_CONFIG:-fl/distributed/app/tiny.toml}
OUT=${OUT:-results_fl/docker_logs}
mkdir -p "$OUT"

[ -d deploy/nodes/SBI ] || python -m fl.distributed.prepare --out deploy/nodes
mkdir -p deploy/models
if [ "$RUN_CONFIG" = "fl/distributed/app/tiny.toml" ] && [ ! -d deploy/models/tiny-qwen2 ]; then
  python -m fl.distributed.tiny_model --data deploy/nodes --out deploy/models/tiny-qwen2
fi

if [ "${LOBO:-0}" = "1" ]; then PROFILE=""; ARM=fedavg_lobo; MIN=2; else PROFILE="--profile all-banks"; ARM=fedavg; MIN=3; fi
COMPOSE="docker compose -f deploy/docker-compose.yml $PROFILE"
$COMPOSE build
$COMPOSE up -d
trap '$COMPOSE logs --no-color > "$OUT/compose_$ARM.log" 2>&1; $COMPOSE down' EXIT
sleep 15

export FLWR_HOME="$REPO/$OUT/flwr_home"; mkdir -p "$FLWR_HOME"
printf '[superlink]\ndefault = "docker"\n[superlink.docker]\naddress = "127.0.0.1:8000"\ninsecure = true\n' > "$FLWR_HOME/config.toml"
MERGED="$OUT/run_config_$ARM.toml"
python - "$RUN_CONFIG" "$MERGED" "$ARM" "$MIN" <<'PY'
import json, sys, tomllib
src, dst, arm, min_nodes = sys.argv[1:]
rc = tomllib.loads(open(src).read())
rc.update({"arm": arm, "min-nodes": int(min_nodes)})   # paths stay relative to /app in the containers
fmt = lambda v: ("true" if v else "false") if isinstance(v, bool) else json.dumps(v)
open(dst, "w").write("".join(f"{k} = {fmt(v)}\n" for k, v in rc.items()))
PY
PYTHONPATH="$REPO" flwr run fl/distributed/app docker --run-config "$MERGED" --stream 2>&1 | tee "$OUT/run_$ARM.log"
sleep 15
echo "Done. Results in results_fl/ (out-dir of $RUN_CONFIG)"
