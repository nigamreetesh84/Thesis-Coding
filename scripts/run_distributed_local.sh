#!/usr/bin/env bash
# Setup A: SuperLink + one SuperNode per bank as separate processes on this machine.
#
#   bash scripts/run_distributed_local.sh                       # tiny offline model (smoke test)
#   RUN_CONFIG=fl/distributed/app/quick.toml bash scripts/run_distributed_local.sh   # D1 (Qwen2.5-0.5B)
#   BANKS="SBI HDFC" MIN_NODES=2 ARM=fedavg_lobo RUN_CONFIG=... bash scripts/run_distributed_local.sh
#   TLS=1 bash scripts/run_distributed_local.sh                 # HTTPS between banks and server
#
# Logs go to $LOG_DIR; all processes are stopped at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

BANKS=${BANKS:-"SBI HDFC AXIS"}
MIN_NODES=${MIN_NODES:-3}
ARM=${ARM:-fedavg}
RUN_CONFIG=${RUN_CONFIG:-fl/distributed/app/tiny.toml}
NODES=${NODES:-deploy/nodes}
LOG_DIR=${LOG_DIR:-results_fl/logs}
mkdir -p "$LOG_DIR"

if [ ! -d "$NODES/SBI" ]; then
  python -m fl.distributed.prepare --out "$NODES" ${SMALL:+--small}
fi
if [ "$RUN_CONFIG" = "fl/distributed/app/tiny.toml" ] && [ ! -d deploy/models/tiny-qwen2 ]; then
  python -m fl.distributed.tiny_model --data "$NODES" --out deploy/models/tiny-qwen2
fi

if [ "${TLS:-0}" = "1" ]; then
  [ -f deploy/certs/server.crt ] || bash deploy/certs/make_certs.sh
  LINK_SEC="--ssl-ca-certfile deploy/certs/ca.crt --ssl-certfile deploy/certs/server.crt --ssl-keyfile deploy/certs/server.key"
  NODE_SEC="--root-certificates deploy/certs/ca.crt"
  FEDERATION=deployment-tls
else
  LINK_SEC="--insecure"; NODE_SEC="--insecure"; FEDERATION=deployment
fi

# Flower CLI connection settings (the CLI needs absolute certificate paths)
export FLWR_HOME="$REPO/$LOG_DIR/flwr_home"
mkdir -p "$FLWR_HOME"
cat > "$FLWR_HOME/config.toml" <<TOML
[superlink]
default = "$FEDERATION"
[superlink.deployment]
address = "127.0.0.1:8000"
insecure = true
[superlink.deployment-tls]
address = "127.0.0.1:8000"
root-certificates = "$REPO/deploy/certs/ca.crt"
TOML

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

rm -f "$LOG_DIR/superlink_state.db"
flower-superlink $LINK_SEC --disable-runtime-dependency-installation \
  --database "$LOG_DIR/superlink_state.db" > "$LOG_DIR/superlink.log" 2>&1 &
PIDS+=($!)
sleep 4

port=9100
for BANK in $BANKS; do
  flower-supernode $NODE_SEC --superlink 127.0.0.1:8000 --port $port \
    --node-config "bank='$BANK' data-dir='$REPO/$NODES/$BANK'" > "$LOG_DIR/node_$BANK.log" 2>&1 &
  PIDS+=($!)
  port=$((port + 1))
done
sleep 8

# flwr does not mix a TOML file with key=value overrides, so write one merged file
MERGED="$LOG_DIR/run_config_$ARM.toml"
python - "$RUN_CONFIG" "$MERGED" "$ARM" "$MIN_NODES" "$REPO/$NODES/server/eval" <<'PY'
import sys, tomllib, json
src, dst, arm, min_nodes, eval_dir = sys.argv[1:]
rc = tomllib.loads(open(src).read())
rc.update({"arm": arm, "min-nodes": int(min_nodes), "eval-dir": eval_dir})
fmt = lambda v: ("true" if v else "false") if isinstance(v, bool) else json.dumps(v)
open(dst, "w").write("".join(f"{k} = {fmt(v)}\n" for k, v in rc.items()))
PY
flwr run fl/distributed/app "$FEDERATION" --run-config "$MERGED" --stream \
  2>&1 | tee "$LOG_DIR/run_$ARM.log"
sleep 10  # the ServerApp finishes writing results after the stream ends
echo "Done. Results: see out-dir in $RUN_CONFIG"
