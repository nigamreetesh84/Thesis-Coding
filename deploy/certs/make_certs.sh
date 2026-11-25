#!/usr/bin/env bash
# Self-signed CA + SuperLink certificate for TLS between the banks and the server.
#   bash deploy/certs/make_certs.sh [extra-hostname-or-ip ...]
# The server certificate is valid for: localhost, superlink (docker service name),
# 127.0.0.1, 10.42.0.10 (network-namespace demo) and any extra names/IPs given.
# Banks only need ca.crt (to verify the server); never copy server.key off the server.
set -euo pipefail
cd "$(dirname "$0")"

SAN="DNS:localhost,DNS:superlink,IP:127.0.0.1,IP:10.42.0.10"
for extra in "$@"; do
  if [[ "$extra" =~ ^[0-9.]+$ ]]; then SAN="$SAN,IP:$extra"; else SAN="$SAN,DNS:$extra"; fi
done

openssl genrsa -out ca.key 4096 2>/dev/null
openssl req -x509 -new -nodes -key ca.key -sha256 -days 365 -subj "/CN=RQ5 FL demo CA" -out ca.crt

openssl genrsa -out server.key 4096 2>/dev/null
openssl req -new -key server.key -subj "/CN=superlink" -out server.csr
printf "subjectAltName=%s\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n" "$SAN" > server.ext
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -sha256 -extfile server.ext
rm -f server.csr server.ext ca.srl
chmod 600 ca.key server.key
echo "Wrote $(pwd)/{ca.crt,ca.key,server.crt,server.key}  SAN=$SAN"
