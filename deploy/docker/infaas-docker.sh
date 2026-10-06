#!/usr/bin/env bash
# INFaaS on docker (one host, one worker container per GPU). See docs/DOCKER.md.
#
#   deploy/docker/infaas-docker.sh gpus                 GPU indices + a suggested GPUS line
#   deploy/docker/infaas-docker.sh build                build controller/worker images
#   deploy/docker/infaas-docker.sh download [--models a,b]   fetch weights into MODEL_DIR
#   deploy/docker/infaas-docker.sh gen                  write generated/docker-compose.yml
#   deploy/docker/infaas-docker.sh up                   gen + start everything, wait for workers
#   deploy/docker/infaas-docker.sh down                 stop and remove the containers
#   deploy/docker/infaas-docker.sh ps | logs [service]  status / logs (docker compose)
#   deploy/docker/infaas-docker.sh state                workers, variants, states (Metadata Store)
#   deploy/docker/infaas-docker.sh register [--all|--models a,b] [--image F]   profile + register
#   deploy/docker/infaas-docker.sh query MODEL IMAGE SLO_MS [-n N]  one request, native API
#
# INFAAS_ENV=<file> selects another settings file (default: infaas-docker.env here).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ENV_FILE="${INFAAS_ENV:-$HERE/infaas-docker.env}"
OUT="$HERE/generated"
COMPOSE=(docker compose -f "$OUT/docker-compose.yml")

get() {  # get KEY [default] — last KEY=VALUE in the settings file
  local v
  v="$(grep -E "^[[:space:]]*$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true)"
  v="${v%\"}"; v="${v#\"}"
  echo "${v:-${2:-}}"
}
CONTROLLER_IMAGE="$(get CONTROLLER_IMAGE infaas-controller:0.1.0)"
WORKER_IMAGE="$(get WORKER_IMAGE infaas-worker:0.1.0)"
NETWORK="$(get NETWORK bridge)"

# how a one-off container reaches the controller
if [ "$NETWORK" = "host" ]; then
  NET_ARGS=(--network host); CTRL=127.0.0.1; REDIS=127.0.0.1
else
  NET_ARGS=(--network infaas); CTRL=infaas-controller; REDIS=infaas-redis
fi

gen() { python3 "$HERE/gen_compose.py" --env "$ENV_FILE" --out "$OUT" "$@"; }

cmd="${1:-help}"; shift || true
case "$cmd" in
  gpus)
    python3 "$HERE/gen_compose.py" --suggest ;;
  build)
    docker build -f "$ROOT/docker/controller.Dockerfile" -t "$CONTROLLER_IMAGE" "$ROOT"
    docker build -f "$ROOT/docker/worker.Dockerfile" -t "$WORKER_IMAGE" "$ROOT" ;;
  download)
    MODEL_DIR="$(get MODEL_DIR /data/podexec-models)"
    mkdir -p "$MODEL_DIR" 2>/dev/null || true
    docker run --rm -v "$MODEL_DIR:/models" -e HF_HOME=/models -e HF_HUB_OFFLINE=0 \
      "$WORKER_IMAGE" python -m infaas.cli.download_models "$@" ;;
  gen)
    gen "$@" ;;
  up)
    gen "$@"
    "${COMPOSE[@]}" up -d --remove-orphans
    echo ">> waiting for workers to report healthy (library preload takes a while)..."
    for _ in $(seq 1 90); do
      total=$("${COMPOSE[@]}" ps --format '{{.Name}}' | grep -c -- '-worker-' || true)
      ok=$("${COMPOSE[@]}" ps --format '{{.Name}} {{.Health}}' | grep -- '-worker-' | grep -c ' healthy' || true)
      echo "   workers healthy: $ok/$total"
      [ "$total" -gt 0 ] && [ "$ok" -eq "$total" ] && break
      sleep 5
    done
    "${COMPOSE[@]}" ps ;;
  down)
    "${COMPOSE[@]}" down --remove-orphans ;;
  ps)
    "${COMPOSE[@]}" ps "$@" ;;
  logs)
    "${COMPOSE[@]}" logs --tail=100 "$@" ;;
  state)
    docker run --rm "${NET_ARGS[@]}" "$CONTROLLER_IMAGE" \
      python -m infaas.cli.state --redis "$REDIS:16379" "$@" ;;
  register)
    # PROFILE_MODE=original needs no image; service needs --image FILE
    vol=(); img_arg=(); args=()
    while [ $# -gt 0 ]; do
      case "$1" in
        --image) vol=(-v "$(realpath "$2"):/in/image.jpg:ro"); img_arg=(--image /in/image.jpg); shift 2 ;;
        *) args+=("$1"); shift ;;
      esac
    done
    [ ${#args[@]} -gt 0 ] || args=(--all)
    docker run --rm "${NET_ARGS[@]}" "${vol[@]}" "$CONTROLLER_IMAGE" \
      python -m infaas.cli.register --controller "$CTRL:50053" "${img_arg[@]}" "${args[@]}" ;;
  query)
    model="${1:?usage: query MODEL IMAGE SLO_MS [-n N]}"; img="${2:?image}"; slo="${3:?slo_ms}"; shift 3
    docker run --rm "${NET_ARGS[@]}" -v "$(realpath "$img"):/in/image.jpg:ro" "$CONTROLLER_IMAGE" \
      python -m infaas.cli.online_query --controller "$CTRL:50052" \
      --model "$model" --image /in/image.jpg --slo "$slo" "$@" ;;
  *)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
