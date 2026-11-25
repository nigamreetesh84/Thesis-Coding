#!/usr/bin/env bash
# Multi-host emulation on ONE Linux machine (needs root + iproute2), no Docker needed.
#
# Every node gets its own network namespace (own network stack and IP address) and
# its own Unix user, and talks to the server only over a virtual Ethernet bridge:
#
#   br-fl 10.42.0.1 ── fl-server 10.42.0.10  SuperLink + ServerApp   (user fl_server)
#                   ├─ fl-sbi    10.42.0.11  SuperNode SBI            (user bank_sbi)
#                   ├─ fl-hdfc   10.42.0.12  SuperNode HDFC           (user bank_hdfc)
#                   └─ fl-axis   10.42.0.13  SuperNode AXIS           (user bank_axis)
#
# Each bank's data folder is owned by that bank's user with mode 700, so a bank
# process cannot even read another bank's file; the server user cannot read any.
# The script records this as evidence in $OUT/isolation_evidence.txt.
#
#   sudo bash scripts/run_distributed_netns.sh                          # tiny offline model
#   sudo RUN_CONFIG=fl/distributed/app/quick.toml bash scripts/run_distributed_netns.sh
#   sudo TLS=1 bash scripts/run_distributed_netns.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
RUN_CONFIG=${RUN_CONFIG:-fl/distributed/app/tiny.toml}
NODES=${NODES:-deploy/nodes}
SRV=${SRV:-/srv/fl}
OUT=${OUT:-$REPO/results_fl/netns}
# never send node-to-node traffic through an HTTP(S) proxy set in the environment
export NO_PROXY="10.42.0.10,10.42.0.11,10.42.0.12,10.42.0.13,localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
declare -A IP=([server]=10.42.0.10 [SBI]=10.42.0.11 [HDFC]=10.42.0.12 [AXIS]=10.42.0.13)
declare -A NS=([server]=fl-server [SBI]=fl-sbi [HDFC]=fl-hdfc [AXIS]=fl-axis)
declare -A USR=([server]=fl_server [SBI]=bank_sbi [HDFC]=bank_hdfc [AXIS]=bank_axis)

[ -d "$NODES/SBI" ] || python -m fl.distributed.prepare --out "$NODES"
if [ "$RUN_CONFIG" = "fl/distributed/app/tiny.toml" ] && [ ! -d deploy/models/tiny-qwen2 ]; then
  python -m fl.distributed.tiny_model --data "$NODES" --out deploy/models/tiny-qwen2
fi

teardown() {
  pkill -f "flower-super" 2>/dev/null || true
  for n in "${NS[@]}"; do ip netns del "$n" 2>/dev/null || true; done
  ip link del br-fl 2>/dev/null || true
  command -v iptables >/dev/null && iptables -D FORWARD -i br-fl -o br-fl -j ACCEPT 2>/dev/null || true
}
trap teardown EXIT
teardown

# --- network: one bridge, one namespace + veth pair per node -------------------
ip link add br-fl type bridge
ip addr add 10.42.0.1/24 dev br-fl
ip link set br-fl up
# hosts with Docker set the FORWARD policy to DROP; allow traffic between the nodes on br-fl
command -v iptables >/dev/null && iptables -C FORWARD -i br-fl -o br-fl -j ACCEPT 2>/dev/null \
  || { command -v iptables >/dev/null && iptables -I FORWARD -i br-fl -o br-fl -j ACCEPT; } || true
for node in server SBI HDFC AXIS; do
  ns=${NS[$node]}; host_if="v-$ns"
  ip netns add "$ns"
  ip link add "$host_if" type veth peer name eth0 netns "$ns"
  ip link set "$host_if" master br-fl up
  ip -n "$ns" addr add "${IP[$node]}/24" dev eth0
  ip -n "$ns" link set eth0 up
  ip -n "$ns" link set lo up
done

# --- users, code and data ------------------------------------------------------
mkdir -p "$SRV" "$OUT"
for node in server SBI HDFC AXIS; do id -u "${USR[$node]}" >/dev/null 2>&1 || useradd -r -M -s /usr/sbin/nologin "${USR[$node]}"; done
rm -rf "$SRV/code"; mkdir -p "$SRV/code"
cp -r "$REPO/fl" "$REPO/src" "$REPO/evaluation" "$REPO/config" "$SRV/code/"
[ -d "$REPO/deploy/models" ] && mkdir -p "$SRV/code/deploy" && cp -r "$REPO/deploy/models" "$SRV/code/deploy/"
chmod -R a+rX "$SRV/code"
for bank in SBI HDFC AXIS; do
  rm -rf "${SRV:?}/$bank"; mkdir -p "$SRV/$bank/home"
  cp "$NODES/$bank/train.jsonl" "$SRV/$bank/"
  chown -R "${USR[$bank]}" "$SRV/$bank"; chmod -R go-rwx "$SRV/$bank"
done
rm -rf "$SRV/server"; mkdir -p "$SRV/server/home" "$SRV/server/results"
cp -r "$NODES/server/eval" "$SRV/server/"
chown -R "${USR[server]}" "$SRV/server"; chmod -R go-rwx "$SRV/server"

run_as() {  # run_as <node> <cmd...>   inside the node's namespace, as the node's user
  local node=$1; shift
  ip netns exec "${NS[$node]}" runuser -u "${USR[$node]}" -- env PYTHONPATH="$SRV/code" \
    HOME="$SRV/$node/home" FLWR_HOME="$SRV/$node/home/.flwr" HF_HOME="${HF_HOME:-$SRV/$node/home/hf}" "$@"
}

# --- TLS -----------------------------------------------------------------------
if [ "${TLS:-0}" = "1" ]; then
  [ -f deploy/certs/server.crt ] || bash deploy/certs/make_certs.sh
  cp deploy/certs/server.crt deploy/certs/server.key deploy/certs/ca.crt "$SRV/server/"
  chown "${USR[server]}" "$SRV/server/"*.crt "$SRV/server/server.key"
  for bank in SBI HDFC AXIS; do cp deploy/certs/ca.crt "$SRV/$bank/"; chown "${USR[$bank]}" "$SRV/$bank/ca.crt"; done
  LINK_SEC="--ssl-ca-certfile $SRV/server/ca.crt --ssl-certfile $SRV/server/server.crt --ssl-keyfile $SRV/server/server.key"
  FED_SEC="root-certificates = \"$REPO/deploy/certs/ca.crt\""
else
  LINK_SEC="--insecure"; FED_SEC="insecure = true"
fi

# --- evidence of isolation -----------------------------------------------------
EV="$OUT/isolation_evidence.txt"
{
  echo "# Isolation evidence ($(date -u +%FT%TZ))"
  for node in server SBI HDFC AXIS; do
    echo "## ${NS[$node]} (user ${USR[$node]})"
    ip -n "${NS[$node]}" -4 -brief addr show | sed 's/^/  /'
  done
  echo "## SBI's process tries to read HDFC's training file:"
  run_as SBI cat "$SRV/HDFC/train.jsonl" 2>&1 | head -1 | sed 's/^/  /' || true
  echo "## The server process tries to read SBI's training file:"
  run_as server cat "$SRV/SBI/train.jsonl" 2>&1 | head -1 | sed 's/^/  /' || true
  echo "## SBI's namespace reaches the server only through the bridge (ping 10.42.0.10):"
  ip netns exec fl-sbi ping -c 1 -W 1 10.42.0.10 2>&1 | grep -E "bytes from|loss" | sed 's/^/  /' || true
} > "$EV"

# --- start the nodes ---------------------------------------------------------
run_as server flower-superlink $LINK_SEC --host 0.0.0.0 --disable-runtime-dependency-installation \
  > "$OUT/superlink.log" 2>&1 &
sleep 5
for bank in SBI HDFC AXIS; do
  sec="--insecure"; [ "${TLS:-0}" = "1" ] && sec="--root-certificates $SRV/$bank/ca.crt"
  run_as "$bank" flower-supernode $sec --superlink "${IP[server]}:8000" \
    --node-config "bank='$bank' data-dir='$SRV/$bank'" > "$OUT/node_$bank.log" 2>&1 &
done
sleep 10

# --- submit the run from the host (researcher's machine) over the bridge -----
export FLWR_HOME="$OUT/flwr_home"; mkdir -p "$FLWR_HOME"
printf '[superlink]\ndefault = "netns"\n[superlink.netns]\naddress = "%s:8000"\n%s\n' "${IP[server]}" "$FED_SEC" > "$FLWR_HOME/config.toml"
MERGED="$OUT/run_config.toml"
python - "$RUN_CONFIG" "$MERGED" "$SRV" <<'PY'
import json, sys, tomllib
src, dst, srv = sys.argv[1:]
rc = tomllib.loads(open(src).read())
if not rc.get("model-name", "").startswith(("Qwen/", "/")):
    rc["model-name"] = f"{srv}/code/{rc['model-name']}"
rc.update({"out-dir": f"{srv}/server/results", "eval-dir": f"{srv}/server/eval"})
fmt = lambda v: ("true" if v else "false") if isinstance(v, bool) else json.dumps(v)
open(dst, "w").write("".join(f"{k} = {fmt(v)}\n" for k, v in rc.items()))
PY
PYTHONPATH="$REPO" flwr run fl/distributed/app netns --run-config "$MERGED" --stream 2>&1 | tee "$OUT/run.log"
sleep 10
cp -r "$SRV/server/results/." "$OUT/"
echo "Done. Results and isolation evidence in $OUT"
